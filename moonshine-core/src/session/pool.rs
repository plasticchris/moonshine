//! Concurrent multi-seat session pool.
//!
//! Hosts up to N simultaneous streaming sessions in a single process, each
//! pinned to a distinct GPU drawn from a pool and bound to OS-ephemeral ports
//! that are reported to the client via the Moonlight protocol (`sessionUrl0` for
//! RTSP, RTSP `SETUP` `server_port` for video/audio/control).
//!
//! Each seat reuses the existing per-session [`SessionManager`] (keeping its
//! state machine and watchdog unchanged) plus a dedicated per-session
//! [`RtspServer`]. A shared HTTP/HTTPS webserver and mDNS responder route
//! clients into the pool by cert fingerprint.
//!
//! ## Modes
//! - Multi-seat: `compositor.gpus` non-empty. Each `/launch` grabs a free GPU
//!   (or the app's `gpu`, treated as a hard constraint) and binds ephemeral ports.
//! - Single-seat (back-compat): `compositor.gpus` empty. Capacity 1, fixed
//!   config ports, GPU resolution left to the session (honouring `compositor.gpu`
//!   and per-app `gpu`).

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use async_shutdown::ShutdownManager;
use tokio::sync::Mutex;

use crate::rtsp::RtspServer;
use crate::session::compositor::{find_render_node, CompositorConfig};
use crate::session::manager::SessionManager;
use crate::session::stream::audio::AudioStreamConfig;
use crate::session::stream::control::ControlStreamConfig;
use crate::session::stream::video::VideoStreamConfig;
use crate::session::{SessionContext, SessionKeyData, APP_LAUNCH_HTTP_TIMEOUT_SECS};
use crate::ShutdownReason;

/// Reason a launch could not be satisfied.
#[derive(Debug, Clone, Copy)]
pub enum LaunchError {
	/// No free GPU (pool at capacity, or the app's requested GPU is busy/unknown).
	NoGpuAvailable,
	/// The application/compositor failed to start.
	Failed,
}

/// Snapshot of a client's active seat, used for per-client `serverinfo`.
#[derive(Debug, Clone, Copy)]
pub struct SeatInfo {
	pub rtsp_port: u16,
	pub application_id: i32,
}

/// A GPU slot in the pool.
struct GpuEntry {
	/// Resolved DRM render node (e.g. `/dev/dri/renderD128`).
	node: PathBuf,
	/// Whether a session currently owns this GPU (reserved or active).
	busy: bool,
}

/// One active streaming seat.
struct Seat {
	gpu_index: Option<usize>,
	owner: String,
	application_id: i32,
	rtsp_port: u16,
	session_manager: SessionManager,
	/// Kept alive for the seat's lifetime; the accept loop is cancelled when the
	/// session's shutdown triggers, releasing the RTSP port.
	_rtsp_server: RtspServer,
}

struct SessionPoolInner {
	base_compositor: CompositorConfig,
	video_config: VideoStreamConfig,
	audio_config: AudioStreamConfig,
	control_config: ControlStreamConfig,
	address: String,
	/// Fixed RTSP port used in single-seat mode.
	rtsp_port: u16,
	stream_timeout: u64,
	shutdown: ShutdownManager<ShutdownReason>,
	/// Multi-seat with ephemeral ports (`compositor.gpus` non-empty).
	ephemeral: bool,
	/// GPU pool (multi-seat only; empty in single-seat mode).
	gpus: Vec<GpuEntry>,
	seats: HashMap<u64, Seat>,
	/// Client identity (cert fingerprint) -> seat id.
	owners: HashMap<String, u64>,
	next_id: u64,
}

impl SessionPoolInner {
	/// Whether a new session can currently be launched.
	fn has_free_slot(&self) -> bool {
		if self.ephemeral {
			self.gpus.iter().any(|g| !g.busy)
		} else {
			self.seats.is_empty()
		}
	}

	/// Reserve a GPU for a new session, honouring an optional per-app hard
	/// constraint. Returns the reserved GPU index (None in single-seat mode).
	fn reserve_gpu(&mut self, app_gpu: &Option<String>) -> Result<Option<usize>, LaunchError> {
		if !self.ephemeral {
			// Single-seat: capacity 1; GPU pinning handled by the session itself.
			return if self.seats.is_empty() {
				Ok(None)
			} else {
				Err(LaunchError::NoGpuAvailable)
			};
		}

		if let Some(selector) = app_gpu {
			// Hard constraint: the app must run on this specific GPU.
			let want = find_render_node(&Some(selector.clone())).map_err(|e| {
				tracing::warn!("Application GPU '{selector}' could not be resolved: {e}");
				LaunchError::NoGpuAvailable
			})?;
			let want = canonical(&want);
			match self.gpus.iter().position(|g| canonical(&g.node) == want) {
				Some(i) if !self.gpus[i].busy => {
					self.gpus[i].busy = true;
					Ok(Some(i))
				},
				Some(_) => {
					tracing::warn!("Application GPU '{selector}' is busy; rejecting launch.");
					Err(LaunchError::NoGpuAvailable)
				},
				None => {
					tracing::warn!("Application GPU '{selector}' is not part of the configured pool.");
					Err(LaunchError::NoGpuAvailable)
				},
			}
		} else {
			match self.gpus.iter().position(|g| !g.busy) {
				Some(i) => {
					self.gpus[i].busy = true;
					Ok(Some(i))
				},
				None => Err(LaunchError::NoGpuAvailable),
			}
		}
	}
}

/// Manages the lifecycle of all concurrent streaming sessions.
#[derive(Clone)]
pub struct SessionPool {
	inner: Arc<Mutex<SessionPoolInner>>,
}

impl SessionPool {
	#[allow(clippy::result_unit_err)]
	#[allow(clippy::too_many_arguments)]
	pub fn new(
		compositor_config: CompositorConfig,
		video_config: VideoStreamConfig,
		audio_config: AudioStreamConfig,
		control_config: ControlStreamConfig,
		address: String,
		rtsp_port: u16,
		stream_timeout: u64,
		shutdown: ShutdownManager<ShutdownReason>,
	) -> Result<Self, ()> {
		let ephemeral = !compositor_config.gpus.is_empty();

		let mut gpus: Vec<GpuEntry> = Vec::new();
		if ephemeral {
			for selector in &compositor_config.gpus {
				let node = find_render_node(&Some(selector.clone())).map_err(|e| {
					tracing::error!("Failed to resolve GPU '{selector}' for the multi-seat pool: {e}");
				})?;
				let node_canon = canonical(&node);
				// Dedupe: distinct GPUs only, so each seat gets its own hardware.
				if gpus.iter().any(|g| canonical(&g.node) == node_canon) {
					tracing::warn!("Ignoring duplicate GPU pool entry '{selector}' ({}).", node.display());
					continue;
				}
				tracing::info!("Multi-seat GPU pool entry: {selector} -> {}", node.display());
				gpus.push(GpuEntry { node, busy: false });
			}
			if gpus.is_empty() {
				tracing::error!("Multi-seat requested but no GPUs resolved from `compositor.gpus`.");
				return Err(());
			}
			tracing::info!(
				"Concurrent multi-seat enabled: {} GPU(s), OS-ephemeral ports.",
				gpus.len()
			);
		} else {
			tracing::info!("Single-seat mode: one session on fixed config ports.");
		}

		Ok(Self {
			inner: Arc::new(Mutex::new(SessionPoolInner {
				base_compositor: compositor_config,
				video_config,
				audio_config,
				control_config,
				address,
				rtsp_port,
				stream_timeout,
				shutdown,
				ephemeral,
				gpus,
				seats: HashMap::new(),
				owners: HashMap::new(),
				next_id: 1,
			})),
		})
	}

	/// Launch a new session for `owner`, returning the per-session RTSP port to
	/// advertise in `sessionUrl0`.
	pub async fn launch(&self, owner: String, context: SessionContext) -> Result<u16, LaunchError> {
		// A client re-launching (e.g. picking a different app) replaces its own
		// session: tear the old one down first so its GPU/ports are freed.
		self.stop_owner(&owner).await;

		let application_id = context.application_id;

		// Reserve a GPU + id and snapshot the per-seat config under the lock,
		// then do the (slow) initialize/launch without holding it.
		let plan = {
			let mut inner = self.inner.lock().await;
			let gpu_index = inner.reserve_gpu(&context.application.gpu)?;
			let gpu_node = gpu_index.map(|i| inner.gpus[i].node.clone());

			let id = inner.next_id;
			inner.next_id += 1;

			// Per-seat isolated HOME (multi-seat only). `{owner}` keeps a client's
			// profile persistent across whatever GPU is free; `{seat}` ties it to
			// the GPU slot instead.
			let seat_home = if inner.ephemeral {
				inner
					.base_compositor
					.seat_home_template
					.as_deref()
					.map(|template| seat_home_path(template, gpu_index, &owner))
			} else {
				None
			};
			// Only relevant alongside a per-seat HOME.
			let steam_libraries = if seat_home.is_some() {
				inner.base_compositor.seat_steam_libraries.clone()
			} else {
				Vec::new()
			};

			let mut compositor_config = inner.base_compositor.clone();
			compositor_config.gpus = Vec::new();
			if let Some(node) = &gpu_node {
				// Pin both compositor and encoder to the reserved GPU.
				compositor_config.gpu = Some(node.to_string_lossy().into_owned());
			}

			let (video_config, audio_config, control_config, rtsp_bind_port) = if inner.ephemeral {
				let mut video = inner.video_config.clone();
				let mut audio = inner.audio_config.clone();
				let mut control = inner.control_config.clone();
				video.port = 0;
				audio.port = 0;
				control.port = 0;
				(video, audio, control, 0)
			} else {
				(
					inner.video_config.clone(),
					inner.audio_config.clone(),
					inner.control_config.clone(),
					inner.rtsp_port,
				)
			};

			SeatPlan {
				id,
				gpu_index,
				seat_home,
				steam_libraries,
				compositor_config,
				video_config,
				audio_config,
				control_config,
				address: inner.address.clone(),
				rtsp_bind_port,
				stream_timeout: inner.stream_timeout,
				shutdown: inner.shutdown.clone(),
			}
		};

		let gpu_index = plan.gpu_index;
		let id = plan.id;

		match self.bring_up_seat(plan, context).await {
			Ok((session_manager, rtsp_server, rtsp_port)) => {
				{
					let mut inner = self.inner.lock().await;
					inner.seats.insert(
						id,
						Seat {
							gpu_index,
							owner: owner.clone(),
							application_id,
							rtsp_port,
							session_manager: session_manager.clone(),
							_rtsp_server: rtsp_server,
						},
					);
					inner.owners.insert(owner, id);
				}
				self.spawn_seat_monitor(id, session_manager);
				Ok(rtsp_port)
			},
			Err(e) => {
				// Release the reserved GPU on failure.
				let mut inner = self.inner.lock().await;
				if let Some(i) = gpu_index {
					if let Some(gpu) = inner.gpus.get_mut(i) {
						gpu.busy = false;
					}
				}
				Err(e)
			},
		}
	}

	/// Create the session manager + RTSP server and launch the app. On any
	/// failure the partially-initialized session is torn down before returning.
	async fn bring_up_seat(
		&self,
		plan: SeatPlan,
		mut context: SessionContext,
	) -> Result<(SessionManager, RtspServer, u16), LaunchError> {
		// Redirect the app's HOME to the seat's isolated profile (multi-seat).
		context.seat_home = plan.seat_home.clone();
		if let Some(home) = &plan.seat_home {
			seed_steam_libraries(home, &plan.steam_libraries);
		}

		let session_manager = SessionManager::new(
			plan.compositor_config,
			plan.video_config.clone(),
			plan.audio_config,
			plan.control_config,
			plan.address.clone(),
			plan.id,
			plan.stream_timeout,
			plan.shutdown,
		)
		.map_err(|()| LaunchError::Failed)?;

		if session_manager.initialize_session(context).await.is_err() {
			return Err(LaunchError::Failed);
		}

		let ports = match session_manager.stream_ports().await {
			Some(ports) => ports,
			None => {
				let _ = session_manager.stop_session().await;
				return Err(LaunchError::Failed);
			},
		};

		// Per-session RTSP server, cancelled when the session's shutdown triggers.
		let stop = session_manager.session_stop().await;
		let (rtsp_server, rtsp_port) = match RtspServer::start(
			plan.address,
			plan.rtsp_bind_port,
			ports,
			plan.video_config,
			session_manager.clone(),
			async move {
				let _ = stop.wait_shutdown_triggered().await;
			},
		)
		.await
		{
			Ok(v) => v,
			Err(()) => {
				let _ = session_manager.stop_session().await;
				return Err(LaunchError::Failed);
			},
		};

		// Launch compositor + application (bounded, matching the old HTTP path).
		match tokio::time::timeout(
			std::time::Duration::from_secs(APP_LAUNCH_HTTP_TIMEOUT_SECS),
			session_manager.launch_session(),
		)
		.await
		{
			Ok(Ok(())) => Ok((session_manager, rtsp_server, rtsp_port)),
			Ok(Err(())) => {
				let _ = session_manager.stop_session().await;
				Err(LaunchError::Failed)
			},
			Err(_) => {
				tracing::error!("Timed out waiting for application launch.");
				let _ = session_manager.stop_session().await;
				Err(LaunchError::Failed)
			},
		}
	}

	/// Refresh the session keys for `owner`'s active seat (client reconnect).
	/// Returns the seat's RTSP port for `sessionUrl0`.
	pub async fn resume(&self, owner: &str, keys: SessionKeyData) -> Result<u16, ()> {
		let (session_manager, rtsp_port) = {
			let inner = self.inner.lock().await;
			let id = inner.owners.get(owner).ok_or(())?;
			let seat = inner.seats.get(id).ok_or(())?;
			(seat.session_manager.clone(), seat.rtsp_port)
		};
		session_manager.update_keys(keys).await?;
		Ok(rtsp_port)
	}

	/// Stop `owner`'s active seat, if any. Returns whether a seat was stopped.
	pub async fn stop_owner(&self, owner: &str) -> bool {
		let seat = {
			let inner = self.inner.lock().await;
			match inner.owners.get(owner) {
				Some(id) => inner.seats.get(id).map(|s| (*id, s.session_manager.clone())),
				None => None,
			}
		};

		match seat {
			Some((id, session_manager)) => {
				// Stop first so the session is fully torn down (`session = None`)
				// before the seat — and its `SessionManager` — is dropped.
				let _ = session_manager.stop_session().await;
				self.remove_seat(id).await;
				true
			},
			None => false,
		}
	}

	/// The active seat owned by `owner`, for per-client `serverinfo`.
	pub async fn seat_info(&self, owner: &str) -> Option<SeatInfo> {
		let inner = self.inner.lock().await;
		let id = inner.owners.get(owner)?;
		let seat = inner.seats.get(id)?;
		Some(SeatInfo {
			rtsp_port: seat.rtsp_port,
			application_id: seat.application_id,
		})
	}

	/// Whether the pool can currently accept a new session.
	pub async fn has_capacity(&self) -> bool {
		self.inner.lock().await.has_free_slot()
	}

	/// Remove a seat and free its GPU + owner mapping. Idempotent: safe to call
	/// from both the end-monitor and an explicit stop.
	async fn remove_seat(&self, id: u64) {
		let seat = {
			let mut inner = self.inner.lock().await;
			let seat = inner.seats.remove(&id);
			if let Some(seat) = &seat {
				if let Some(i) = seat.gpu_index {
					if let Some(gpu) = inner.gpus.get_mut(i) {
						gpu.busy = false;
					}
				}
				inner.owners.remove(&seat.owner);
			}
			seat
		};
		// Drop the seat (SessionManager + RtspServer) outside the lock. The
		// session is already stopped, so SessionManager::drop is a no-op.
		drop(seat);
	}

	/// Watch a seat's session and free its slot when the session ends (either by
	/// user cancel or the application exiting).
	fn spawn_seat_monitor(&self, id: u64, session_manager: SessionManager) {
		let pool = self.clone();
		tokio::spawn(async move {
			session_manager.wait_session_end().await;
			tracing::info!("Session {id} ended; freeing GPU/port slot.");
			pool.remove_seat(id).await;
		});
	}
}

/// Per-seat launch plan captured under the pool lock, then executed without it.
struct SeatPlan {
	id: u64,
	gpu_index: Option<usize>,
	/// Per-seat isolated HOME (multi-seat only), derived from the GPU slot.
	seat_home: Option<PathBuf>,
	/// Shared Steam library folders to seed into a fresh seat HOME.
	steam_libraries: Vec<String>,
	compositor_config: CompositorConfig,
	video_config: VideoStreamConfig,
	audio_config: AudioStreamConfig,
	control_config: ControlStreamConfig,
	address: String,
	rtsp_bind_port: u16,
	stream_timeout: u64,
	shutdown: ShutdownManager<ShutdownReason>,
}

/// Canonicalize a render-node path for comparison, falling back to the input on error.
fn canonical(path: &std::path::Path) -> PathBuf {
	std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf())
}

/// Expand a seat HOME template. `{owner}` -> sanitized client identity (stable
/// per client, so a person keeps their profile whichever GPU is free); `{seat}`
/// -> GPU pool slot (hardware-tied).
fn seat_home_path(template: &str, gpu_index: Option<usize>, owner: &str) -> PathBuf {
	let seat = gpu_index.map(|i| i.to_string()).unwrap_or_else(|| "single".to_string());
	PathBuf::from(
		template
			.replace("{seat}", &seat)
			.replace("{owner}", &sanitize_owner(owner)),
	)
}

/// Seed a fresh seat's Steam `libraryfolders.vdf` so it already knows about
/// shared library folders (e.g. a system SteamLibrary), letting installed games
/// launch without re-downloading. Only writes when the file is absent, so we
/// never clobber the config Steam evolves and rewrites after its first run.
fn seed_steam_libraries(seat_home: &std::path::Path, libraries: &[String]) {
	let libraries: Vec<&String> = libraries.iter().filter(|p| std::path::Path::new(p).is_dir()).collect();
	if libraries.is_empty() {
		return;
	}

	let steam_dir = seat_home.join(".local/share/Steam");

	// Index 0 is the seat's own install library; the shared folders follow.
	// Minimal path-only entries; Steam fills in contentid/size/apps on first run.
	let mut vdf = String::from("\"libraryfolders\"\n{\n");
	vdf.push_str(&format!("\t\"0\"\n\t{{\n\t\t\"path\"\t\t\"{}\"\n\t}}\n", steam_dir.display()));
	for (i, lib) in libraries.iter().enumerate() {
		vdf.push_str(&format!("\t\"{}\"\n\t{{\n\t\t\"path\"\t\t\"{lib}\"\n\t}}\n", i + 1));
	}
	vdf.push_str("}\n");

	let mut seeded = false;
	for rel in ["steamapps", "config"] {
		let dir = steam_dir.join(rel);
		let file = dir.join("libraryfolders.vdf");
		if file.exists() {
			continue;
		}
		if let Err(e) = std::fs::create_dir_all(&dir) {
			tracing::warn!("Failed to create {} for Steam library seed: {e}", dir.display());
			continue;
		}
		match std::fs::write(&file, &vdf) {
			Ok(()) => seeded = true,
			Err(e) => tracing::warn!("Failed to seed Steam libraries at {}: {e}", file.display()),
		}
	}
	if seeded {
		tracing::info!(
			target: "onboarding",
			"Registered {} shared Steam library folder(s) in new seat {}",
			libraries.len(),
			steam_dir.display()
		);
	}
}

/// Filesystem-safe, stable directory component from a seat owner (a configured
/// user name, or a raw cert fingerprint / `uniqueid` when unmapped). Keeps
/// alphanumerics plus `-`, `_`, `.` so friendly user names stay readable; falls
/// back to "default".
fn sanitize_owner(owner: &str) -> String {
	let s: String = owner
		.chars()
		.filter(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
		.collect();
	if s.is_empty() {
		"default".to_string()
	} else {
		s
	}
}
