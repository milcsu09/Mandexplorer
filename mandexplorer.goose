import std;
import os;
import math;
import gfx;
import ui;
import vec;

// Mandexplorer ported to Goose. Rendering runs in a persistent worker pool;
// the display thread never blocks on a render.

struct QV { pos: float2, uv: float2 }
struct ViewP { p: float4 }

const LN2: f64 = 0.6931471805599453;

// Adaptive iteration control — 1:1 port of the C controller.
const AUTO_BUDGET_MS: f64 = 6000.0;
const AUTO_TARGET_MS: f64 = 4000.0;
const AUTO_HARD_MS: f64 = 60000.0;
const AUTO_SENSITIVITY_LO: f64 = 0.005;
const AUTO_SENSITIVITY_HI: f64 = 0.020;
const AUTO_DEEP_WIDTH: f64 = 1e-12;
const MAX_ITER_MAX: u64 = 16777216;
const PROBE_COLS: i64 = 64;
const PROBE_ROWS: i64 = 48;
const PROBE_DEADLINE_MS: f64 = 200.0;
const PROBE_MIN_SAMPLES: i64 = 512;

// Vibe renders are deliberately incomplete.  The display keeps zooming by
// 1.02 every frame, so a frame that took 15 frames to finish is already
// magnified ~1.35x and its centre is a blurry, pixelated mess.  Restart from
// the cursor as soon as the inner half of the disk is fresh (or the zoom has
// drifted too far), instead of grinding out the invisible outer rim.  Old
// jobs die by generation; new ones start at the centre of the distance-sorted
// list, so the visible part is always the newest pixels.
const VIBE_STOP_FRAC: f64 = 0.5;
const VIBE_MAX_DRIFT: f64 = 1.02;
// The visible centre is a tiny fraction of the 250px disk.  Waiting for 50%
// of the disk means ~10 frames with no fresh centre pixels, which is exactly
// the blurry mush.  Refresh the inner ~8% as soon as it lands instead.
const VIBE_CENTER_CHUNK: i64 = 256;
const VIBE_CENTER_TARGET: i64 = 16384;
// Never abort a live render just because the zoom drifted: the old code did
// that every ~0.3s on a deep view, so workers were killed before their first
// 256-pixel batch arrived and the centre stayed a magnified soap bubble for
// 10-15 seconds.  Only a stuck render (or no progress at all) gets killed.
const VIBE_HARD_DRIFT: f64 = 5.0;

// auto_state values (AUTO_OFF, AUTO_HOLD, ... AUTO_VIBE).
const AS_OFF: i32 = 0; const AS_HOLD: i32 = 1; const AS_GROW: i32 = 2;
const AS_CONVERGED: i32 = 3; const AS_BUDGET: i32 = 4; const AS_RESCUE: i32 = 5;
const AS_DEEP: i32 = 6; const AS_VIBE: i32 = 7;

const FORMULA_MANDEL: i32 = 0;
const FORMULA_OMNI: i32 = 1;

struct Palette { size: i64, data: u32[>..] }

// Flat job sent to a worker. Orbit and palette are copied per job so workers
// never share mutable state with the display thread.
struct Job {
    gen: u64, w: i64, h: i64, ss: i64,
    center_re: f64, center_im: f64, width: f64, bailout: f64,
    iter_max: u64, iter_per_color: u64,
    formula: i32, omni_a: f64, lx: f64, ly: f64, lz: f64,
    color_inside: i32,
    row_start: i64, row_count: i64,
    indices: i32[],
    is_probe: bool, probe_iter2: u64, fit_mode: bool, fit_attempt: i64,
    probe_deadline: f64, probe_min_samples: i64,
    mouse_cx: i64, mouse_cy: i64, vibe_render: bool,
    delta_c_re: f64, delta_c_im: f64, period_tol: f64,
    palette_size: i64, palette_data: u32[],
    orbit_c_re: f64, orbit_c_im: f64
}

// A worker owns its reference orbit.  Copying an iter_max-sized orbit into
// every job was tens/hundreds of MB per frame at deep zoom -- the real reason
// vibe showed seconds of soap.  The orbit is cheap to recompute (one f64
// loop) and stays local to the worker program.
root struct WorkerOrbit {
    c_re: f64, c_im: f64, n: u64, valid: bool,
    re: f64[>..], im: f64[>..]
}

fn ensure_orbit(job: Job&, wo: WorkerOrbit&) {
    if wo.valid && wo.n == job.iter_max
        && wo.c_re == job.orbit_c_re && wo.c_im == job.orbit_c_im { return; }
    let n = job.iter_max + 1;
    wo.re.resize(n, 0.0);
    wo.im.resize(n, 0.0);
    var zr = 0.0; var zi = 0.0;
    for i in n {
        wo.re[i] = zr; wo.im[i] = zi;
        let nzr = zr * zr - zi * zi + job.orbit_c_re;
        zi = 2.0 * zr * zi + job.orbit_c_im;
        zr = nzr;
    }
    wo.c_re = job.orbit_c_re; wo.c_im = job.orbit_c_im;
    wo.n = job.iter_max; wo.valid = true;
}

// Flat result: a batch of pixels rendered in distance-from-mouse order.
// The main thread scatters them into the displayed image so the fresh
// pixels expand as a circle around the cursor (as in the original C build).
struct PixResult { gen: u64, indices: i32[], colors: u32[] }
struct ProbeResult { gen: u64, changed: i64, total: i64, fit: bool, attempt: i64 }
struct RowJob { y: i64, cnt: i64, r2: f64 }

root struct AppState {
    // committed scene view (what the in-flight render is for)
    c_re: f64, c_im: f64, view_width: f64,
    bailout: f64, lx: f64, ly: f64, lz: f64,
    iter_max: u64, iter_per_color: u64,
    formula: i32, omni_a: f64, color_inside: i32,
    // display transform: maps the last complete frame to the current view
    disp_scale: f64, disp_pan_x: f64, disp_pan_y: f64,
    // scene the in-flight render was dispatched for, and the display
    // transform at that moment (used to rebase on completion)
    pend_re: f64, pend_im: f64, pend_width: f64, pend_valid: bool,
    snap_scale: f64, snap_pan_x: f64, snap_pan_y: f64,
    w: i64, h: i64, ss: i64,
    image: u32[>..],        // last complete frame, displayed through disp transform
    render_buf: u32[>..],   // buffer filled by the in-flight render
    orbit_re: f64[>..], orbit_im: f64[>..],
    orbit_valid: bool, orbit_iter_max: u64, orbit_c_re: f64, orbit_c_im: f64,
    delta_c_re: f64, delta_c_im: f64,
    palette: Palette, palette_path: u8[>..],
    gen: u64, rendering: bool, need_redraw: bool,
    pixels_done: i64, pixels_total: i64,
    mouse_cx: i64, mouse_cy: i64, vibe_since_dispatch: i64,
    show_info: bool, show_help: bool,
    auto_iter: bool, auto_state: i32, auto_deep_zoom: bool, vibe: bool,
    auto_render_ms: f64, auto_sensitivity: f64,
    probe_changed: i64, probe_total: i64, probe_pending: bool, probe_width: f64,
    render_start: f64, last_render_ms: f64,
    period_tol: f64, recording: bool,
    tex_bytes: u8[>..],
    tex: gfx::Texture, worker_ids: i64[>..]
}

fn complex_exp(re: f64, im: f64) -> f64, f64 {
    if abs(re) + abs(im) < 0.25 {
        let w2_re = re * re - im * im;
        let w2_im = 2.0 * re * im;
        let w3_re = w2_re * re - w2_im * im;
        let w3_im = w2_re * im + w2_im * re;
        let w4_re = w3_re * re - w3_im * im;
        let w4_im = w3_re * im + w3_im * re;
        let w5_re = w4_re * re - w4_im * im;
        let w5_im = w4_re * im + w4_im * re;
        let w6_re = w5_re * re - w5_im * im;
        let w6_im = w5_re * im + w5_im * re;
        let out_re = 1.0 + re + w2_re / 2.0 + w3_re / 6.0 + w4_re / 24.0 + w5_re / 120.0 + w6_re / 720.0;
        let out_im = im + w2_im / 2.0 + w3_im / 6.0 + w4_im / 24.0 + w5_im / 120.0 + w6_im / 720.0;
        return out_re, out_im;
    }
    let e = exp(re);
    return e * cos(im), e * sin(im);
}

fn compute_orbit(app: AppState&) {
    if app.formula == FORMULA_OMNI { app.orbit_valid = true; return; }
    app.orbit_re.resize(app.iter_max + 1, 0.0);
    app.orbit_im.resize(app.iter_max + 1, 0.0);
    var zr = 0.0; var zi = 0.0;
    for i in app.iter_max + 1 {
        app.orbit_re[i] = zr;
        app.orbit_im[i] = zi;
        let nzr = zr * zr - zi * zi + app.c_re;
        zi = 2.0 * zr * zi + app.c_im;
        zr = nzr;
    }
    app.orbit_valid = true;
    app.orbit_iter_max = app.iter_max;
    app.orbit_c_re = app.c_re;
    app.orbit_c_im = app.c_im;
    app.delta_c_re = 0.0;
    app.delta_c_im = 0.0;
}

// Exact C semantics: reuse the reference orbit when the center is unchanged,
// or when the change is tiny relative to the viewport width.  Recomputing a
// high-precision orbit is the expensive part of every dispatch, so this is
// the single biggest CPU win outside of the GPU path.
fn orbit_cache_reuse(app: AppState&) -> bool {
    if !app.orbit_valid { return false; }
    if app.formula == FORMULA_OMNI { return false; }
    if app.iter_max != app.orbit_iter_max { return false; }
    let dc_re = app.c_re - app.orbit_c_re;
    let dc_im = app.c_im - app.orbit_c_im;
    if dc_re == 0.0 && dc_im == 0.0 {
        app.delta_c_re = 0.0; app.delta_c_im = 0.0;
        return true;
    }
    if app.view_width > 1e-10
        && abs(dc_re) / app.view_width < 1e-3
        && abs(dc_im) / app.view_width < 1e-3 {
        app.delta_c_re = dc_re; app.delta_c_im = dc_im;
        return true;
    }
    return false;
}

// Perturbation iteration with period detection. Returns smooth iteration
// count nu and the derivative normal (nx, ny) used for shading.
fn mandel(app: AppState&, orbit_re: f64[>..]&, orbit_im: f64[>..]&,
          delta_re: f64, delta_im: f64) -> f64, f64, f64 {
    if app.formula == FORMULA_OMNI {
        let a = app.omni_a;
        var z_re = a / 2.0; var z_im = 0.0;
        var deriv_re = 0.0; var deriv_im = 0.0;
        let c_re = app.c_re + delta_re;
        let c_im = app.c_im + delta_im;
        let bailout2 = app.bailout * app.bailout;
        for iter in app.iter_max {
            let den = z_re * z_re + z_im * z_im;
            if !(den > 0.0) { return iter as f64, 0.0, 0.0; }
            let w_re = a * z_re / den;
            let w_im = -a * z_im / den;
            let e_re, e_im = complex_exp(w_re, w_im);
            let z2_re = z_re * z_re - z_im * z_im;
            let z2_im = 2.0 * z_re * z_im;
            let g_re = z2_re * e_re - z2_im * e_im;
            let g_im = z2_re * e_im + z2_im * e_re;
            let gp_re = e_re * (2.0 * z_re - a) - e_im * (2.0 * z_im);
            let gp_im = e_re * (2.0 * z_im) + e_im * (2.0 * z_re - a);
            let nd_re = gp_re * deriv_re - gp_im * deriv_im + 1.0;
            let nd_im = gp_re * deriv_im + gp_im * deriv_re;
            deriv_re = nd_re; deriv_im = nd_im;
            z_re = g_re + c_re; z_im = g_im + c_im;
            let zm2 = z_re * z_re + z_im * z_im;
            if !is_finite(zm2) { return iter as f64, 0.0, 0.0; }
            if zm2 > bailout2 {
                let aa = z_re * deriv_re + z_im * deriv_im;
                let bb = z_im * deriv_re - z_re * deriv_im;
                let len = hypot(aa, bb);
                var nx = 0.0; var ny = 0.0;
                if len > 0.0 { nx = aa / len; ny = bb / len; }
                let nu = (iter as f64) + 1.0 - log(log(zm2) * 0.5) / LN2;
                return nu, nx, ny;
            }
        }
        return app.iter_max as f64, 0.0, 0.0;
    }

    let x = app.c_re + delta_re;
    let y = app.c_im + delta_im;
    let dx2 = x + 1.0;
    if dx2 * dx2 + y * y <= 1.0 / 16.0 { return app.iter_max as f64, 0.0, 0.0; }
    let qx = x - 0.25;
    let q = qx * qx + y * y;
    if q * (q + qx) <= y * y / 4.0 { return app.iter_max as f64, 0.0, 0.0; }

    let d_re = delta_re + app.delta_c_re;
    let d_im = delta_im + app.delta_c_im;

    var iter: u64 = 0;
    var io: u64 = 0;
    var period: u64 = 1;
    var saved_re = 0.0; var saved_im = 0.0; var have_saved = false;
    var eps_re = 0.0; var eps_im = 0.0;
    var deriv_re = 0.0; var deriv_im = 0.0;
    let bailout2 = app.bailout * app.bailout;

    while iter < app.iter_max {
        let r_re = orbit_re[io];
        let r_im = orbit_im[io];
        let z_re = r_re + eps_re;
        let z_im = r_im + eps_im;

        let ndr = 2.0 * (z_re * deriv_re - z_im * deriv_im) + 1.0;
        let ndi = 2.0 * (z_re * deriv_im + z_im * deriv_re);
        deriv_re = ndr; deriv_im = ndi;

        let lin_re = 2.0 * (r_re * eps_re - r_im * eps_im);
        let lin_im = 2.0 * (r_re * eps_im + r_im * eps_re);
        let quad_re = eps_re * eps_re - eps_im * eps_im;
        let quad_im = 2.0 * eps_re * eps_im;
        eps_re = lin_re + quad_re + d_re;
        eps_im = lin_im + quad_im + d_im;

        let r1_re = orbit_re[io + 1];
        let r1_im = orbit_im[io + 1];
        let z1_re = r1_re + eps_re;
        let z1_im = r1_im + eps_im;
        let zm2 = z1_re * z1_re + z1_im * z1_im;

        if have_saved {
            let dr = z1_re - saved_re;
            let di = z1_im - saved_im;
            if app.period_tol > 1e-61 && dr * dr + di * di < app.period_tol * (1.0 + zm2) {
                return app.iter_max as f64, 0.0, 0.0;
            }
        }

        if iter + 1 == period {
            saved_re = z1_re; saved_im = z1_im; have_saved = true;
            period *= 2;
            if period > app.iter_max { period = app.iter_max; }
        }

        if zm2 > bailout2 {
            let aa = z1_re * deriv_re + z1_im * deriv_im;
            let bb = z1_im * deriv_re - z1_re * deriv_im;
            let len = hypot(aa, bb);
            var nx = 0.0; var ny = 0.0;
            if len > 0.0 { nx = aa / len; ny = bb / len; }
            let nu = max(0.0, (iter as f64) + 1.0 - log(log(zm2) * 0.5) / LN2);
            return nu, nx, ny;
        }

        iter += 1;
        io += 1;
        if eps_re * eps_re + eps_im * eps_im > zm2 / 2.0 {
            eps_re = z1_re; eps_im = z1_im; io = 0;
        }
    }
    return app.iter_max as f64, 0.0, 0.0;
}

fn pal_get(pal: Palette&, i: i64) -> u32 {
    let n = pal.size;
    if n <= 0 { return 0xFF000000; }
    var k = i % n;
    if k < 0 { k = k + n; }
    return pal.data[k];
}

fn pixel_color(app: AppState&, orbit_re: f64[>..]&, orbit_im: f64[>..]&,
               dx: f64, dy: f64) -> u32 {
    let nu, nx, ny = mandel(app, orbit_re, orbit_im, dx, dy);
    if nu >= app.iter_max as f64 {
        if app.color_inside == 1 && app.view_width <= 1e-3 { return 0xFF181430; }
        return 0xFF000000;
    }
    let t = nu / (app.iter_per_color as f64);
    let i = t as i64;
    let f = t - (i as f64);
    let c1 = pal_get(app.palette, i);
    let c2 = pal_get(app.palette, i + 1);
    let r0 = (c1 >> 16) & 0xFF; let g0 = (c1 >> 8) & 0xFF; let b0 = c1 & 0xFF;
    let r1 = (c2 >> 16) & 0xFF; let g1 = (c2 >> 8) & 0xFF; let b1 = c2 & 0xFF;
    var r = (r0 as f64) + ((r1 as f64) - (r0 as f64)) * f;
    var g = (g0 as f64) + ((g1 as f64) - (g0 as f64)) * f;
    var b = (b0 as f64) + ((b1 as f64) - (b0 as f64)) * f;
    let shade_raw = (nx * app.lx + ny * app.ly + app.lz) / sqrt(nx * nx + ny * ny + 1.0);
    let shade = max(0.0, shade_raw * 0.7 + 0.3);
    r = r * shade; g = g * shade; b = b * shade;
    return (0xFF000000 as u32) | ((r as i64) as u32 << 16) | ((g as i64) as u32 << 8) | ((b as i64) as u32);
}

// Render a job's rows, but walk the pixels in increasing distance from the
// mouse and stream them back in small batches.  The display thread scatters
// each batch into the visible image, producing the expanding circle.
fn auto_state_name(st: i32) -> u8[:] {
    if st == AS_OFF { return "off"; }
    if st == AS_HOLD { return "hold"; }
    if st == AS_GROW { return "grow"; }
    if st == AS_CONVERGED { return "converged"; }
    if st == AS_BUDGET { return "budget"; }
    if st == AS_RESCUE { return "rescue"; }
    if st == AS_DEEP { return "deep-zoom"; }
    if st == AS_VIBE { return "vibe"; }
    return "hold";
}

fn probe_known_interior(x: f64, y: f64) -> bool {
    let dx = x + 1.0;
    if dx * dx + y * y <= 1.0 / 16.0 { return true; }
    let qx = x - 0.25;
    let q = qx * qx + y * y;
    if q * (q + qx) <= y * y / 4.0 { return true; }
    return false;
}

// Returns (escaped, nu).  Same plain-f64 iteration as the C probe — the
// perturbation orbit is deliberately not used here.
fn probe_mandel(cre: f64, cim: f64, max_iter: u64, bailout2: f64) -> bool, f64 {
    var zr = 0.0; var zi = 0.0;
    var iter: u64 = 0;
    while iter < max_iter {
        let zr2 = zr * zr; let zi2 = zi * zi;
        zi = 2.0 * zr * zi + cim;
        zr = zr2 - zi2 + cre;
        let mag2 = zr * zr + zi * zi;
        if mag2 > bailout2 {
            let nu = (iter as f64) + 1.0 - log(log(mag2) * 0.5) / LN2;
            return true, max(0.0, nu);
        }
        iter += 1;
    }
    return false, max_iter as f64;
}

fn probe_omni(cre: f64, cim: f64, max_iter: u64, bailout2: f64, a: f64) -> bool, f64 {
    var zr = a / 2.0; var zi = 0.0;
    var iter: u64 = 0;
    while iter < max_iter {
        let den = zr * zr + zi * zi;
        if !(den > 0.0) { return true, iter as f64; }
        let wre = a * zr / den;
        let wim = -a * zi / den;
        let er, ei = complex_exp(wre, wim);
        let z2r = zr * zr - zi * zi;
        let z2i = 2.0 * zr * zi;
        zr = z2r * er - z2i * ei + cre;
        zi = z2r * ei + z2i * er + cim;
        let mag2 = zr * zr + zi * zi;
        if !is_finite(mag2) { return true, iter as f64; }
        if mag2 > bailout2 { return true, (iter + 1) as f64; }
        iter += 1;
    }
    return false, max_iter as f64;
}

fn probe_color(pal: Palette&, ipc: i64, nu: f64) -> u32 {
    let p = if ipc > 0 { ipc as f64 } else { 1.0 };
    let t = nu / p;
    let i = t as i64;
    let f = t - (i as f64);
    let c1 = pal_get(pal, i);
    let c2 = pal_get(pal, i + 1);
    let r0 = ((c1 >> 16) & 0xFF) as f64; let g0 = ((c1 >> 8) & 0xFF) as f64; let b0 = (c1 & 0xFF) as f64;
    let r1 = ((c2 >> 16) & 0xFF) as f64; let g1 = ((c2 >> 8) & 0xFF) as f64; let b1 = (c2 & 0xFF) as f64;
    return (0xFF000000 as u32)
        | (((r0 + (r1-r0)*f) as i64) as u32 << 16)
        | (((g0 + (g1-g0)*f) as i64) as u32 << 8)
        | (((b0 + (b1-b0)*f) as i64) as u32);
}

fn probe_colors_differ(pal: Palette&, ipc: i64, a: f64, b: f64) -> bool {
    let ca = probe_color(pal, ipc, a);
    let cb = probe_color(pal, ipc, b);
    let dr = max(((ca >> 16) & 0xFF) as i64 - ((cb >> 16) & 0xFF) as i64,
                  ((cb >> 16) & 0xFF) as i64 - ((ca >> 16) & 0xFF) as i64);
    let dg = max(((ca >> 8) & 0xFF) as i64 - ((cb >> 8) & 0xFF) as i64,
                  ((cb >> 8) & 0xFF) as i64 - ((ca >> 8) & 0xFF) as i64);
    let db = max((ca & 0xFF) as i64 - (cb & 0xFF) as i64,
                  (cb & 0xFF) as i64 - (ca & 0xFF) as i64);
    return max(dr, max(dg, db)) >= 2;
}

fn probe_work(job: Job&) {
    var pal = Palette { size: job.palette_size, data: copy(job.palette_data) };
    let n = if job.iter_max > 0 { job.iter_max } else { 1 };
    let n2 = if n > MAX_ITER_MAX / 4 { MAX_ITER_MAX } else { n * 4 };
    let aspect = (job.h as f64) / (job.w as f64);
    let b2 = job.bailout * job.bailout;
    let start = clock();
    var changed: i64 = 0; var total: i64 = 0;
    var row: i64 = 0;
    while row < PROBE_ROWS {
        let gy = (row * 17) % PROBE_ROWS;
        var gx: i64 = 0;
        while gx < PROBE_COLS {
            if clock() - start >= job.probe_deadline / 1000.0 { row = PROBE_ROWS; break; }
            let dx = ((gx as f64 + 0.5) / (PROBE_COLS as f64) - 0.5) * job.width;
            let dy = (0.5 - (gy as f64 + 0.5) / (PROBE_ROWS as f64)) * job.width * aspect;
            let cre = job.center_re + dx;
            let cim = job.center_im + dy;
            var hit = false;
            if job.formula == FORMULA_OMNI {
                let a1, n1 = probe_omni(cre, cim, n, b2, job.omni_a);
                let a2, n2v = probe_omni(cre, cim, n2, b2, job.omni_a);
                hit = if a1 != a2 { true } else { if !a1 { false } else { probe_colors_differ(pal, job.iter_per_color as i64, n1, n2v) } };
            } else {
                if probe_known_interior(cre, cim) { total += 1; gx += 1; continue; }
                let a1, n1 = probe_mandel(cre, cim, n, b2);
                let a2, n2v = probe_mandel(cre, cim, n2, b2);
                hit = if a1 != a2 { true } else { if !a1 { false } else { probe_colors_differ(pal, job.iter_per_color as i64, n1, n2v) } };
            }
            total += 1;
            if hit { changed += 1; }
            gx += 1;
        }
        row += 1;
    }
    if total < job.probe_min_samples { changed = 0; total = 0; }
    qput(ProbeResult { gen: job.gen, changed: changed, total: total,
                       fit: job.fit_mode, attempt: job.fit_attempt });
}

fn fit_video_n(app: AppState&, v: ViewState&, n: u64, gen: u64, attempt: i64) {
    var min_n = n / 64;
    let color_floor = v.iter_per_color * 4;
    if min_n < color_floor { min_n = color_floor; }
    if min_n < 1 { min_n = 1; }
    qput(Job {
        gen: gen, w: app.w, h: app.h, ss: 1,
        center_re: v.c_re, center_im: v.c_im, width: v.width,
        bailout: v.bailout, iter_max: n, iter_per_color: v.iter_per_color,
        formula: v.formula, omni_a: v.omni_a, lx: v.lx, ly: v.ly, lz: v.lz,
        color_inside: if v.color_inside { 1 } else { 0 },
        row_start: -2, row_count: 0, indices: [],
        is_probe: true, probe_iter2: 0, fit_mode: true, fit_attempt: attempt,
        probe_deadline: 500.0, probe_min_samples: 128,
        mouse_cx: app.mouse_cx, mouse_cy: app.mouse_cy, vibe_render: false,
        delta_c_re: 0.0, delta_c_im: 0.0, period_tol: app.period_tol,
        palette_size: app.palette.size, palette_data: copy(app.palette.data),
        orbit_c_re: 0.0, orbit_c_im: 0.0
    });
}

fn auto_iter_update(app: AppState&, render_ms: f64) {
    app.auto_render_ms = render_ms;
    if !app.auto_iter { app.auto_state = AS_OFF; return; }
    if app.vibe { app.auto_state = AS_VIBE; return; }
    if !(app.probe_width > AUTO_DEEP_WIDTH) || !is_finite(app.probe_width) {
        app.auto_state = AS_DEEP; return;
    }
    if app.probe_total == 0 { app.auto_state = AS_HOLD; return; }
    let sensitivity = (app.probe_changed as f64) / (app.probe_total as f64);
    app.auto_sensitivity = sensitivity;
    if sensitivity < AUTO_SENSITIVITY_LO { app.auto_state = AS_CONVERGED; return; }
    let frame_ms = max(render_ms, 1.0);
    var factor = AUTO_TARGET_MS / frame_ms;
    factor = clamp(factor, 1.25, 2.0);
    if sensitivity < AUTO_SENSITIVITY_HI { factor = min(factor, 1.5); }
    var next = (app.iter_max as f64 * factor) as u64;
    if next <= app.iter_max { next = app.iter_max + 1; }
    if next > MAX_ITER_MAX { next = MAX_ITER_MAX; }
    if next == app.iter_max { app.auto_state = AS_BUDGET; return; }
    var estimate = frame_ms * ((next as f64) / (app.iter_max as f64)) * 1.10;
    var rescue = false;
    if estimate > AUTO_BUDGET_MS {
        rescue = true;
        next = (app.iter_max as f64 * 2.0) as u64;
        if next <= app.iter_max { next = app.iter_max + 1; }
        if next > MAX_ITER_MAX { next = MAX_ITER_MAX; }
        estimate = frame_ms * ((next as f64) / (app.iter_max as f64)) * 1.10;
        if estimate > AUTO_HARD_MS { app.auto_state = AS_BUDGET; return; }
    }
    app.iter_max = next;
    app.orbit_valid = false;
    app.need_redraw = true;
    app.auto_state = if rescue { AS_RESCUE } else { AS_GROW };
}

// Render exactly the pixel indices handed to us.  The display thread builds
// a single globally distance-sorted list, so workers consume it in order and
// the visible fill is a true expanding circle (no horizontal band stripes).
fn render_chunk(job: Job&, wo: WorkerOrbit&) {
    ensure_orbit(job, wo);
    var pa = AppState {
        c_re: job.center_re, c_im: job.center_im, view_width: job.width,
        bailout: job.bailout, lx: job.lx, ly: job.ly, lz: job.lz,
        iter_max: job.iter_max, iter_per_color: job.iter_per_color,
        formula: job.formula, omni_a: job.omni_a, color_inside: job.color_inside,
        disp_scale: 1.0, disp_pan_x: 0.0, disp_pan_y: 0.0,
        pend_re: 0.0, pend_im: 0.0, pend_width: job.width, pend_valid: false,
        snap_scale: 1.0, snap_pan_x: 0.0, snap_pan_y: 0.0,
        w: job.w, h: job.h, ss: job.ss, image: [], render_buf: [],
        orbit_re: [], orbit_im: [], orbit_valid: false, orbit_iter_max: job.iter_max,
        orbit_c_re: job.orbit_c_re, orbit_c_im: job.orbit_c_im,
        delta_c_re: job.delta_c_re, delta_c_im: job.delta_c_im,
        palette: Palette { size: job.palette_size, data: copy(job.palette_data) },
        palette_path: [],
        gen: job.gen, rendering: false, need_redraw: false,
        pixels_done: 0, pixels_total: 0, mouse_cx: job.mouse_cx, mouse_cy: job.mouse_cy,
        vibe_since_dispatch: 0,
        show_info: false, show_help: false,
        auto_iter: false, auto_state: 0, auto_deep_zoom: false, vibe: false,
        auto_render_ms: 0.0, auto_sensitivity: 0.0,
        probe_changed: 0, probe_total: 0, probe_pending: false, probe_width: 0.0,
        render_start: 0.0, last_render_ms: 0.0,
        period_tol: job.period_tol, recording: false,
        tex_bytes: [],
        tex: gfx::Texture { id: 0 }, worker_ids: []
    };

    let ss = job.ss;
    var idxs: i32[>..] = [];
    var cols: u32[>..] = [];
    for k in job.indices.len {
        let idx = job.indices[k];
        let x = idx % job.w;
        let y = idx / job.w;
        var r: u32 = 0; var g: u32 = 0; var b: u32 = 0;
        for sy in ss {
            for sx in ss {
                let fx = (x as f64) + (sx as f64 + 0.5) / (ss as f64);
                let fy = (y as f64) + (sy as f64 + 0.5) / (ss as f64);
                let dx = (fx - (job.w as f64) / 2.0) * job.width / (job.w as f64);
                let dy = ((job.h as f64) / 2.0 - fy) * job.width / (job.w as f64);
                let col = pixel_color(pa, wo.re, wo.im, dx, dy);
                r += (col >> 16) & 0xFF;
                g += (col >> 8) & 0xFF;
                b += col & 0xFF;
            }
        }
        let n = (ss * ss) as u32;
        idxs.push(idx);
        cols.push((((r / n) & 0xFF) << 16) | (((g / n) & 0xFF) << 8) | ((b / n) & 0xFF) | (0xFF000000 as u32));
        if idxs.len >= 256 {
            qput(PixResult { gen: job.gen, indices: copy(idxs), colors: copy(cols) });
            idxs.clear(); cols.clear();
        }
    }
    if idxs.len > 0 {
        qput(PixResult { gen: job.gen, indices: copy(idxs), colors: copy(cols) });
    }
}

thread_fn worker() {
    var wo = WorkerOrbit { re: [], im: [], c_re: 0.0, c_im: 0.0, n: 0, valid: false };
    // C's terminate() clears the pool queue on a view change.  Goose has no
    // queue-clear, so each worker remembers the newest generation it has seen
    // and drops older jobs immediately instead of rendering pixels whose
    // results the display thread will discard anyway.
    var seen_gen: u64 = 0;
    loop {
        let job = qget<Job>();
        if job.is_probe { probe_work(job); seen_gen = max(seen_gen, job.gen); continue; }
        if job.row_start < 0 { break; }
        if job.gen < seen_gen { continue; }
        seen_gen = job.gen;
        render_chunk(job, wo);
    }
}

fn load_scene(path: const u8[:], app: AppState&) -> bool {
    var data: u8[>..] = [];
    if !read_file(path, data) { return false; }
    let lines = split(data, '\n');
    var idx = 0;
    if lines.len < 1 { return false; }
    let v, okv = parse_int(trim(lines[0]));
    if !okv || v != 1 { return false; }
    idx = 1;
    let cr, okr = parse_flt(trim(lines[idx])); idx++; if !okr { return false; }
    let ci, oki = parse_flt(trim(lines[idx])); idx++; if !oki { return false; }
    let wv, okw = parse_flt(trim(lines[idx])); idx++; if !okw { return false; }
    let bv, okb = parse_flt(trim(lines[idx])); idx++; if !okb { return false; }
    let lx, okx = parse_flt(trim(lines[idx])); idx++; if !okx { return false; }
    let ly, oky = parse_flt(trim(lines[idx])); idx++; if !oky { return false; }
    let lz, okz = parse_flt(trim(lines[idx])); idx++; if !okz { return false; }
    let im, okim = parse_int(trim(lines[idx])); idx++; if !okim { return false; }
    let ipc, okipc = parse_int(trim(lines[idx])); idx++; if !okipc { return false; }
    if idx >= lines.len { return false; }
    var pp: u8[>..] = [];
    pp.append(trim(lines[idx]));
    app.c_re = cr; app.c_im = ci; app.view_width = wv; app.bailout = bv;
    app.lx = lx; app.ly = ly; app.lz = lz;
    app.iter_max = im as u64; app.iter_per_color = ipc as u64;
    app.palette_path = copy(pp);
    return true;
}

fn load_palette(path: const u8[:], pal: Palette&) -> bool {
    var data: u8[>..] = [];
    if !read_file(path, data) { return false; }
    let lines = split(data, '\n');
    if lines.len < 2 { return false; }
    let v, okv = parse_int(trim(lines[0]));
    if !okv || v != 1 { return false; }
    let sz, oksz = parse_int(trim(lines[1]));
    if !oksz { return false; }
    var arr: u32[>..] = [];
    for i in sz {
        if 2 + i >= lines.len { break; }
        let parts = split(trim(lines[2 + i]), ' ');
        if parts.len < 4 { continue; }
        let a, _ = parse_int(trim(parts[0]));
        let r, _ = parse_int(trim(parts[1]));
        let g, _ = parse_int(trim(parts[2]));
        let b, _ = parse_int(trim(parts[3]));
        arr.push(((a as u32) << 24) | ((r as u32) << 16) | ((g as u32) << 8) | (b as u32));
    }
    pal.size = arr.len;
    pal.data = copy(arr);
    return true;
}

fn save_scene(path: const u8[:], app: AppState&) -> bool {
    var out: u8[>..] = [];
    format(out, "1\n");
    format_flt(out, app.c_re, 17); format(out, "\n");
    format_flt(out, app.c_im, 17); format(out, "\n");
    format_flt(out, app.view_width, 17); format(out, "\n");
    format_flt(out, app.bailout, 17); format(out, "\n");
    format_flt(out, app.lx, 17); format(out, "\n");
    format_flt(out, app.ly, 17); format(out, "\n");
    format_flt(out, app.lz, 17); format(out, "\n");
    format(out, app.iter_max, "\n");
    format(out, app.iter_per_color, "\n");
    format(out, app.palette_path, "\n");
    return write_file_atomic(path, out);
}

fn enqueue_jobs(app: AppState&) {
    // One GLOBAL ordering of all pixels by distance from the mouse, built
    // with a counting sort (O(n)).  Workers get consecutive slices of it, so
    // results come back as true concentric circles instead of row stripes.
    // At deep zoom a full frame costs seconds (1.6B iterations at 8k).  The
    // user watches a small area around the cursor, so render just that and
    // freeze the zoom while it is being computed.  No magnified soap frame.
    let maxr2 = if app.vibe { 80 * 80 }
                else { (app.w / 2) * (app.w / 2) + (app.h / 2) * (app.h / 2) };
    var counts: i64[>..] = [];
    counts.resize(maxr2 + 1, 0);
    var n: i64 = 0;
    for y in app.h {
        let dy = (y as f64) - (app.mouse_cy as f64);
        for x in app.w {
            let dx = (x as f64) - (app.mouse_cx as f64);
            let r2 = (dx * dx + dy * dy) as i64;
            if r2 <= maxr2 { counts[r2] = counts[r2] + 1; n = n + 1; }
        }
    }
    var sum: i64 = 0;
    for r in counts.len {
        let c = counts[r];
        counts[r] = sum;
        sum = sum + c;
    }
    var sorted: i32[>..] = [];
    sorted.resize(n, 0);
    for y in app.h {
        let dy = (y as f64) - (app.mouse_cy as f64);
        for x in app.w {
            let dx = (x as f64) - (app.mouse_cx as f64);
            let r2 = (dx * dx + dy * dy) as i64;
            if r2 <= maxr2 {
                sorted[counts[r2]] = (y * app.w + x) as i32;
                counts[r2] = counts[r2] + 1;
            }
        }
    }

    app.pixels_total = n;
    // The first jobs are the pixels nearest the cursor.  Make them small so
    // the centre lands within a frame or two; the far rim can use big chunks.
    let first_zone = if app.vibe { min(n, VIBE_CENTER_TARGET) } else { 0 };
    var i: i64 = 0;
    while i < n {
        let chunk = if app.vibe {
            if i < first_zone { VIBE_CENTER_CHUNK } else { 2048 }
        } else { 4096 };
        let cnt = if i + chunk < n { chunk } else { n - i };
        var idxs: i32[>..] = [];
        idxs.resize(cnt, 0);
        for k in cnt { idxs[k] = sorted[i + k]; }
        qput(Job {
            gen: app.gen, w: app.w, h: app.h, ss: app.ss,
            center_re: app.c_re, center_im: app.c_im, width: app.view_width,
            bailout: app.bailout, iter_max: app.iter_max, iter_per_color: app.iter_per_color,
            formula: app.formula, omni_a: app.omni_a, lx: app.lx, ly: app.ly, lz: app.lz,
            color_inside: app.color_inside, row_start: i, row_count: cnt,
            indices: copy(idxs), is_probe: false, probe_iter2: 0, fit_mode: false, fit_attempt: 0, probe_deadline: 200.0, probe_min_samples: 512,
            mouse_cx: app.mouse_cx, mouse_cy: app.mouse_cy, vibe_render: app.vibe,
            delta_c_re: app.delta_c_re, delta_c_im: app.delta_c_im, period_tol: app.period_tol,
            palette_size: app.palette.size, palette_data: copy(app.palette.data),
            orbit_c_re: app.orbit_c_re, orbit_c_im: app.orbit_c_im
        });
        i = i + cnt;
    }
}

fn bilerp(c00: u32, c10: u32, c01: u32, c11: u32, tx: f64, ty: f64) -> u32 {
    fn ch(c: u32, sh: i32) -> f64 { return ((c >> sh) & 0xFF) as f64; }
    let r = (ch(c00,16)*(1.0-tx) + ch(c10,16)*tx)*(1.0-ty) + (ch(c01,16)*(1.0-tx) + ch(c11,16)*tx)*ty;
    let g = (ch(c00,8)*(1.0-tx) + ch(c10,8)*tx)*(1.0-ty) + (ch(c01,8)*(1.0-tx) + ch(c11,8)*tx)*ty;
    let b = (ch(c00,0)*(1.0-tx) + ch(c10,0)*tx)*(1.0-ty) + (ch(c01,0)*(1.0-tx) + ch(c11,0)*tx)*ty;
    return (0xFF000000 as u32) | ((r as i64) as u32 << 16) | ((g as i64) as u32 << 8) | ((b as i64) as u32);
}

fn bake_display(app: AppState&) {
    if app.disp_scale == 1.0 && app.disp_pan_x == 0.0 && app.disp_pan_y == 0.0 { return; }
    var out: u32[>..] = [];
    out.resize(app.w * app.h, 0xFF000000 as u32);
    let sc = app.disp_scale;
    let px = app.disp_pan_x / (app.w as f64);
    let py = app.disp_pan_y / (app.h as f64);
    for sy in app.h {
        let vy = (((sy as f64 + 0.5) / (app.h as f64) - 0.5) / sc + 0.5 + py) * (app.h as f64) - 0.5;
        let y0 = clamp(vy as i64, 0, app.h - 1);
        let y1 = clamp(y0 + 1, 0, app.h - 1);
        let ty = clamp(vy - (y0 as f64), 0.0, 1.0);
        for sx in app.w {
            let vx = (((sx as f64 + 0.5) / (app.w as f64) - 0.5) / sc + 0.5 + px) * (app.w as f64) - 0.5;
            let x0 = clamp(vx as i64, 0, app.w - 1);
            let x1 = clamp(x0 + 1, 0, app.w - 1);
            let tx = clamp(vx - (x0 as f64), 0.0, 1.0);
            out[sy * app.w + sx] = bilerp(
                app.image[y0 * app.w + x0], app.image[y0 * app.w + x1],
                app.image[y1 * app.w + x0], app.image[y1 * app.w + x1], tx, ty);
        }
    }
    app.image = copy(out);
    app.disp_scale = 1.0;
    app.disp_pan_x = 0.0;
    app.disp_pan_y = 0.0;
}

fn dispatch_render(app: AppState&) {
    app.vibe_since_dispatch = 0;
    // Bake the current display transform into both the scene and the image.
    // From here on the committed scene and the displayed pixels agree, and
    // disp starts at identity: new pixels can be written straight into image.
    let w = app.w as f64;
    app.c_re += app.disp_pan_x * app.view_width / w;
    app.c_im -= app.disp_pan_y * app.view_width / w;
    app.view_width /= app.disp_scale;
    bake_display(app);

    let mp = gfx::mouse_pos();
    app.mouse_cx = mp.x as i64;
    app.mouse_cy = mp.y as i64;

    app.gen = app.gen + 1;
    // Goose has no qclear.  New jobs append to the same FIFO queue, so
    // without draining, workers grind through all the old generation's jobs
    // before they ever see the new one -- that was the multi-second delay
    // after pressing v.  Main can receive from the queue too, so drain it.
    var dropping = true;
    while dropping {
        let old_job, ok = qpoll<Job>();
        if !ok { dropping = false; }
    }
    if !orbit_cache_reuse(app) { compute_orbit(app); }
    app.render_buf.resize(app.w * app.h, 0xFF000000 as u32);
    for i in app.w * app.h { app.render_buf[i] = app.image[i]; }
    enqueue_jobs(app);
    app.pixels_done = 0;
    app.rendering = true;
    app.need_redraw = false;
    app.render_start = clock();
    app.probe_width = app.view_width;
    app.probe_pending = false;
}

// Bake the accumulated vibe display transform into BOTH the committed scene
// and the displayed pixels.  Resetting disp without rebaking the image makes
// the screen snap back to the old (unzoomed) frame -- exactly the zoom-back
// zig-zag.  This keeps the screen pixel-identical while disp becomes identity.
fn bake_view(app: AppState&) {
    app.pend_valid = false;
    let w = app.w as f64;
    app.c_re += app.disp_pan_x * app.view_width / w;
    app.c_im -= app.disp_pan_y * app.view_width / w;
    app.view_width /= app.disp_scale;
    bake_display(app); // resamples app.image with the same transform as the shader
    // In-flight results were computed in the OLD view's coordinate system.
    // Without a generation bump they would be scattered into the freshly
    // rebaked image and corrupt exactly the pixels we are zooming into.
    app.gen = app.gen + 1;
    app.rendering = false;
    app.pixels_done = 0;
    app.need_redraw = true;
    app.orbit_valid = false;
}

struct ViewState {
    c_re: f64, c_im: f64, width: f64,
    iter_max: u64, recorded_iter_max: u64, iter_per_color: u64,
    ss: i64, formula: i32, omni_a: f64, bailout: f64,
    lx: f64, ly: f64, lz: f64, color_inside: bool
}

fn make_view(app: AppState&) -> ViewState {
    // Effective view includes the live display transform, so recording a
    // vibe zoom captures exactly what is on screen.
    let w = app.w as f64;
    return ViewState {
        c_re: app.c_re + app.disp_pan_x * app.view_width / w,
        c_im: app.c_im - app.disp_pan_y * app.view_width / w,
        width: app.view_width / app.disp_scale,
        iter_max: app.iter_max, recorded_iter_max: app.iter_max,
        iter_per_color: app.iter_per_color,
        ss: app.ss, formula: app.formula, omni_a: app.omni_a,
        bailout: app.bailout, lx: app.lx, ly: app.ly, lz: app.lz,
        color_inside: app.color_inside == 1
    };
}

fn same_view(a: ViewState&, b: ViewState&) -> bool {
    return a.iter_max == b.iter_max && a.ss == b.ss && a.formula == b.formula
        && abs(a.c_re - b.c_re) <= 1e-12 * (abs(b.c_re) + 1.0)
        && abs(a.c_im - b.c_im) <= 1e-12 * (abs(b.c_im) + 1.0)
        && abs(a.width - b.width) <= 1e-12 * (b.width + 1.0);
}

// C's can_reuse_keyframe: a previous frame may be warped instead of rendered
// when the view is a modest zoom of it and all render parameters match.
fn can_reuse_keyframe(kv: ViewState&, v: ViewState&, w: i64, h: i64,
                      idx: i64, kf_idx: i64) -> bool {
    if kv.formula != v.formula { return false; }
    if kv.recorded_iter_max != v.recorded_iter_max { return false; }
    if kv.iter_per_color != v.iter_per_color { return false; }
    if kv.ss != v.ss { return false; }
    if kv.color_inside != v.color_inside { return false; }
    if kv.omni_a != v.omni_a || kv.bailout != v.bailout { return false; }
    if kv.lx != v.lx || kv.ly != v.ly || kv.lz != v.lz { return false; }
    let S = v.width / kv.width;
    let Dx = v.c_re - kv.c_re;
    let Dy = v.c_im - kv.c_im;
    let cx = (w as f64 / 2.0) * (1.0 - S) + Dx * (w as f64) / kv.width;
    let cy = (h as f64 / 2.0) * (1.0 - S) - Dy * (w as f64) / kv.width;
    if cx < 0.0 || (w as f64) * S + cx > (w as f64) - 1.0 { return false; }
    if cy < 0.0 || (h as f64) * S + cy > (h as f64) - 1.0 { return false; }
    if S < 1.0 / 1.5 { return false; }
    if idx - kf_idx >= 5 { return false; }
    return true;
}

fn sample_bilinear(img: u32[>..]&, x: f64, y: f64, w: i64, h: i64) -> u32 {
    var fx = clamp(x, 0.0, w as f64 - 1.0);
    var fy = clamp(y, 0.0, h as f64 - 1.0);
    let x0 = fx as i64; let y0 = fy as i64;
    let x1 = min(x0 + 1, w - 1); let y1 = min(y0 + 1, h - 1);
    let tx = fx - (x0 as f64); let ty = fy - (y0 as f64);
    return bilerp(img[y0 * w + x0], img[y1 * w + x0],
                  img[y0 * w + x1], img[y1 * w + x1], tx, ty);
}

fn warp_keyframe(kv: ViewState&, v: ViewState&, img: u32[>..]&,
                 key: u32[>..]&, w: i64, h: i64) {
    let S = v.width / kv.width;
    let Dx = v.c_re - kv.c_re;
    let Dy = v.c_im - kv.c_im;
    let cx = (w as f64 / 2.0) * (1.0 - S) + Dx * (w as f64) / kv.width;
    let cy = (h as f64 / 2.0) * (1.0 - S) - Dy * (w as f64) / kv.width;
    for y in h {
        let ky = (y as f64) * S + cy;
        for x in w {
            img[y * w + x] = sample_bilinear(key, (x as f64) * S + cx, ky, w, h);
        }
    }
}

// Write one rgb24 frame to a recording file without ever re-reading it.
fn append_frame(path: const u8[:], img: u32[>..]&, buf: u8[>..]&) {
    buf.resize(img.len * 3, 0);
    for i in img.len {
        let c = img[i];
        buf[i * 3] = ((c >> 16) & 0xFF) as u8;
        buf[i * 3 + 1] = ((c >> 8) & 0xFF) as u8;
        buf[i * 3 + 2] = (c & 0xFF) as u8;
    }
    append_file(path, buf);
}

fn add2(out: u8[>..]&, v: i64) {
    out.push((48 + v / 10) as u8);
    out.push((48 + v % 10) as u8);
}

// Same YYYYMMDD_HHMMSS naming as the C version (UTC instead of localtime).
fn timestamped_name(ext: const u8[:]) -> u8[>..] {
    let t = time() as i64;
    let days = t / 86400;
    let sod = t - days * 86400;
    let z = days + 719468;
    let era = if z >= 0 { z / 146097 } else { (z - 146096) / 146097 };
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    var y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    if m <= 2 { y += 1; }
    var out: u8[>..] = [];
    format(out, y, "");
    add2(out, m); add2(out, d);
    out.push('_' as u8);
    add2(out, sod / 3600); add2(out, (sod / 60) % 60); add2(out, sod % 60);
    out.append(ext);
    return out;
}

fn print_help() {
    write_stdout("USAGE: mandexplorer [OPTION]... <SCENE>\n\nOPTION:\n  -w                <int>        Set the window's initial width in pixels\n  -h                <int>        Set the window's initial height in pixels\n  --re              <float>      Set the real part of C\n  --im              <float>      Set the imaginary part of C\n  --width           <float>      Set the viewport width\n  --bailout         <float>      Set the bailout radius\n  --light-angle     <float>      Set the light's angle\n  --iter-max        <int>        Set the maximum number of iterations\n  --iter-per-color  <int>        Set the iterations per color\n  --formula         <NAME>       Formula: mandelbrot or omnibrot\n  --a               <float>      Set omnibrot's 'a' parameter (default 0.005)\n  --palette         <PALETTE>    Set the palette's path\n  --frames          <int>        Exit after N frames (headless smoke test)\n\nSCENE:\n  A '*.sce' file\n\nPALETTE:\n  A '*.pal' file\n");
}
fn next_arg(argv: u8[:][>..]&, i: i64, name: const u8[:]) -> bool {
    if i + 1 >= argv.len {
        var m: u8[>..] = [];
        format(m, "Error: '", name, "' requires a value\n"); write_stderr(m);
        return false;
    }
    return true;
}

fn main() {
    let argv = args();
    var scene_path: u8[>..] = [];
    var frame_limit = -1;
    var opt_w = 800; var opt_h = 600;
    var o_re = 0.0; var o_im = 0.0; var o_width = 0.0; var o_bailout = 0.0;
    var o_angle = 0.0; var o_iter = 0; var o_ipc = 0; var o_a = 0.005;
    var o_formula = -1; var o_palette: u8[>..] = [];
    var h_re = false; var h_im = false; var h_width = false; var h_bailout = false;
    var h_angle = false; var h_iter = false; var h_ipc = false; var h_a = false;
    var gpu_info = false;
    var ai = 1;
    while ai < argv.len {
        let a = argv[ai];
        if a == "--help" { print_help(); return; }
        if a == "-w" || a == "-h" || a == "--re" || a == "--im" || a == "--width"
            || a == "--bailout" || a == "--light-angle" || a == "--iter-max"
            || a == "--iter-per-color" || a == "--formula" || a == "--a"
            || a == "--palette" || a == "--frames" {
            if !next_arg(argv, ai, a) { return; }
            let v = argv[ai + 1];
            if a == "-w" { let n, ok = parse_int(v); if ok { opt_w = n as i64; } }
            else if a == "-h" { let n, ok = parse_int(v); if ok { opt_h = n as i64; } }
            else if a == "--re" { let f, ok = parse_flt(v); if ok { o_re = f; h_re = true; } }
            else if a == "--im" { let f, ok = parse_flt(v); if ok { o_im = f; h_im = true; } }
            else if a == "--width" { let f, ok = parse_flt(v); if ok { o_width = f; h_width = true; } }
            else if a == "--bailout" { let f, ok = parse_flt(v); if ok { o_bailout = f; h_bailout = true; } }
            else if a == "--light-angle" { let f, ok = parse_flt(v); if ok { o_angle = f; h_angle = true; } }
            else if a == "--iter-max" { let n, ok = parse_int(v); if ok { o_iter = n; h_iter = true; } }
            else if a == "--iter-per-color" { let n, ok = parse_int(v); if ok { o_ipc = n; h_ipc = true; } }
            else if a == "--formula" {
                if v == "mandelbrot" { o_formula = FORMULA_MANDEL; }
                else if v == "omnibrot" { o_formula = FORMULA_OMNI; }
                else { write_stderr(str("Error: unknown formula\n")); return; }
            }
            else if a == "--a" { let f, ok = parse_flt(v); if ok { o_a = f; h_a = true; } }
            else if a == "--palette" { o_palette.append(v); }
            else if a == "--frames" { let n, ok = parse_int(v); if ok { frame_limit = n; } }
            ai += 2;
            continue;
        }
        if a == "--no-gpu" { ai += 1; continue; }
        if a == "--gpu-info" {
            write_stdout("GPU available: no (Goose compute shaders have no double precision)\n");
            gpu_info = true;
            ai += 1;
            continue;
        }
        scene_path.append(a);
        ai += 1;
    }
    if gpu_info { return; }
    if scene_path.len == 0 { scene_path.append("default.sce"); }

    var app = AppState {
        c_re: 0.0, c_im: 0.0, view_width: 8.0,
        bailout: 2.0, lx: 0.577, ly: 0.577, lz: 0.577,
        iter_max: 1024, iter_per_color: 16,
        formula: FORMULA_MANDEL, omni_a: 0.005, color_inside: 0,
        disp_scale: 1.0, disp_pan_x: 0.0, disp_pan_y: 0.0,
        pend_re: 0.0, pend_im: 0.0, pend_width: 8.0, pend_valid: false,
        snap_scale: 1.0, snap_pan_x: 0.0, snap_pan_y: 0.0,
        w: opt_w, h: opt_h, ss: 1, image: [], render_buf: [],
        orbit_re: [], orbit_im: [], orbit_valid: false,
        orbit_iter_max: 0, orbit_c_re: 0.0, orbit_c_im: 0.0,
        delta_c_re: 0.0, delta_c_im: 0.0,
        palette: Palette { size: 0, data: [] }, palette_path: [],
        gen: 0, rendering: false, need_redraw: false,
        pixels_done: 0, pixels_total: 0, mouse_cx: 400, mouse_cy: 300, vibe_since_dispatch: 0,
        show_info: true, show_help: false,
        auto_iter: true, auto_state: AS_HOLD, auto_deep_zoom: false, vibe: false,
        auto_render_ms: 0.0, auto_sensitivity: 0.0,
        probe_changed: 0, probe_total: 0, probe_pending: false, probe_width: 0.0,
        render_start: 0.0, last_render_ms: 0.0,
        period_tol: 1e-40, recording: false,
        tex_bytes: [],
        tex: gfx::Texture { id: 0 }, worker_ids: []
    };

    if !load_scene(scene_path, app) {
        write_stderr(str("Failed to load scene\n"));
        return;
    }
    // CLI overrides apply after the scene, exactly like the C version.
    if h_re { app.c_re = o_re; }
    if h_im { app.c_im = o_im; }
    if h_width { app.view_width = o_width; }
    if h_bailout { app.bailout = o_bailout; }
    if h_iter { app.iter_max = o_iter as u64; }
    if h_ipc { app.iter_per_color = o_ipc as u64; }
    if h_a { app.omni_a = o_a; }
    if o_formula >= 0 { app.formula = o_formula as i32; }
    if h_angle {
        let sa = sin(o_angle * 3.141592653589793 / 180.0);
        let ca = cos(o_angle * 3.141592653589793 / 180.0);
        let ll = sqrt(sa * sa + ca * ca + 1.0);
        app.lx = sa / ll; app.ly = ca / ll; app.lz = 1.0 / ll;
    }
    var pal_path: u8[>..] = [];
    if o_palette.len > 0 { pal_path.append(o_palette); }
    else if app.palette_path.len > 0 { pal_path.append(app.palette_path); }
    else { pal_path.append("default.pal"); }
    if !load_palette(pal_path, app.palette) {
        write_stderr(str("Failed to load palette\n"));
        return;
    }

    app.image.resize(app.w * app.h, 0xFF000000 as u32);
    app.render_buf.resize(app.w * app.h, 0);

    guard gfx::open("Mandexplorer", app.w as i64, app.h as i64, gfx::WINDOW_RESIZABLE) else {
        write_stderr(str("gfx open failed: ", gfx::error(), "\n"));
        return;
    }
    defer gfx::close();

    app.tex = gfx::texture2d(app.w, app.h);
    app.tex_bytes.resize(app.w * app.h * 4, 0);
    if app.tex.id == 0 {
        write_stderr(str("texture fail\n"));
        return;
    }
    let nearest = gfx::sampler(gfx::LINEAR, gfx::CLAMP);

    // Fullscreen quad. uv.y = 0 at screen top, 1 at bottom (matches image rows).
    var qverts: QV[>..] = [];
    qverts.push(QV { pos: float2 { -1.0, -1.0 }, uv: float2 { 0.0, 1.0 } });
    qverts.push(QV { pos: float2 { 3.0, -1.0 }, uv: float2 { 2.0, 1.0 } });
    qverts.push(QV { pos: float2 { -1.0, 3.0 }, uv: float2 { 0.0, -1.0 } });
    let qvb = gfx::vertex_buffer(bytes_of(qverts));
    let quad_vs = embed_shader("vert", """
        layout(location = 0) in vec2 a_pos;
        layout(location = 1) in vec2 a_uv;
        layout(location = 0) out vec2 v_uv;
        void main() {
            gl_Position = vec4(a_pos, 0.0, 1.0);
            v_uv = a_uv;
        }
        """);
    // Display transform: p.x = scale, p.y = pan x (normalized), p.z = pan y.
    let quad_fs = embed_shader("frag", """
        layout(set = 2, binding = 0) uniform sampler2D tex;
        layout(set = 3, binding = 0) uniform View { vec4 p; };
        layout(location = 0) in vec2 v_uv;
        layout(location = 0) out vec4 result;
        void main() {
            vec2 uv = (v_uv - vec2(0.5)) / p.x + vec2(0.5) + p.yz;
            result = texture(tex, uv);
        }
        """);
    var qdesc = gfx::PipelineDesc { };
    qdesc.formats[0] = gfx::VERTEX_FLOAT2;
    qdesc.formats[1] = gfx::VERTEX_FLOAT2;
    let quad_pipeline = gfx::pipeline(quad_vs, quad_fs, qdesc);

    let ROBOTO: const u8[:] = embed_file("/home/et/Applications/Mandexplorer/goose-port/goose/samples/data/Roboto-Regular.ttf");
    var ctx = ui::create_from_memory(ROBOTO, 16.0);
    var has_ui = ctx.id != 0;

    // Persistent worker pool, spawned once.
    var nworkers = hardware_threads();
    if nworkers < 1 { nworkers = 1; }
    if nworkers > 8 { nworkers = 8; }
    for i in nworkers { app.worker_ids.push(thread_spawn(worker)); }

    dispatch_render(app);

    var ss_prev = app.ss;
    var drawn = 0;
    // Recording stores view states (like C) and renders them afterwards with
    // keyframe reprojection.  Goose v1 has no subprocesses, so the result is
    // an rgb24 stream instead of an ffmpeg-produced mp4.
    var recorded: ViewState[>..] = [];
    var video_path: u8[>..] = [];
    var video_mode = false;
    var video_index: i64 = 0;
    var video_dispatch_index: i64 = 0;
    var video_pending = false;
    var kf_pending = false;
    var has_keyframe = false;
    var keyframe_index: i64 = -1;
    var keyframe_view = ViewState {
        c_re: 0.0, c_im: 0.0, width: 1.0, iter_max: 1, recorded_iter_max: 1,
        iter_per_color: 1, ss: 1, formula: 0, omni_a: 0.0, bailout: 2.0,
        lx: 0.0, ly: 0.0, lz: 1.0, color_inside: false
    };
    var keyframe_image: u32[>..] = [];
    var frame_bytes: u8[>..] = [];
    var video_fit_pending = false;
    var video_fit_n: u64 = 0;
    var video_fit_attempt: i64 = 0;

    while gfx::frame() {
        let sz = gfx::screen_size();
        if sz.x as i64 != app.w || sz.y as i64 != app.h {
            app.w = sz.x as i64;
            app.h = sz.y as i64;
            app.image.resize(app.w * app.h, 0xFF000000 as u32);
            app.render_buf.resize(app.w * app.h, 0);
            app.tex = gfx::texture2d(app.w, app.h);
            app.tex_bytes.resize(app.w * app.h * 4, 0);
            app.need_redraw = true;
        }

        // ---- input ----
        if gfx::key_pressed("Escape") { break; }
        if !video_mode {
        if gfx::key_pressed("Left Alt") { app.show_info = !app.show_info; }
        if gfx::key_pressed("F1") { app.show_help = !app.show_help; }
        if gfx::key_pressed("h") { app.show_help = !app.show_help; }
        if gfx::key_pressed("x") {
            if app.vibe { bake_view(app); }
            app.c_re = -0.75; app.c_im = 0.0; app.view_width = 8.0;
            app.disp_scale = 1.0; app.disp_pan_x = 0.0; app.disp_pan_y = 0.0;
            app.orbit_valid = false; app.need_redraw = true;
        }
        if gfx::key_pressed("c") {
            if app.vibe { bake_view(app); }
            let mp = gfx::mouse_pos();
            app.c_re += (mp.x as f64 - app.w as f64 / 2.0) * app.view_width / (app.w as f64);
            app.c_im -= (mp.y as f64 - app.h as f64 / 2.0) * app.view_width / (app.w as f64);
            app.orbit_valid = false; app.need_redraw = true;
        }
        if gfx::key_pressed("b") { app.color_inside = if app.color_inside == 1 { 0 } else { 1 }; app.need_redraw = true; }
        if gfx::key_pressed("m") { app.formula = FORMULA_MANDEL; app.orbit_valid = false; app.need_redraw = true; }
        if gfx::key_pressed("o") { app.formula = FORMULA_OMNI; app.orbit_valid = false; app.need_redraw = true; }
        if gfx::key_pressed("i") {
            app.auto_iter = !app.auto_iter;
            app.auto_state = if app.auto_iter { AS_HOLD } else { AS_OFF };
            app.need_redraw = true;
        }
        if gfx::key_pressed("v") {
            // Do NOT bake/reset the display transform here.  `image` still
            // holds the last complete frame in the pre-transform scene, so
            // resetting disp to identity would snap the screen back to that
            // old frame until a fresh render finishes (the "zig-zag").
            // Freezing disp keeps the current view pixel-identical; the
            // completion rebase folds it into the next committed scene.
            app.vibe = !app.vibe;
            if app.vibe {
                ss_prev = app.ss; app.ss = 1;
                // Kill the in-flight full-quality render.  At deep zoom it
                // costs seconds; waiting for it is the "4 seconds of soap"
                // before the first vibe frame.  The small cursor render is
                // dispatched on the next frame instead.
                app.gen = app.gen + 1;
                app.rendering = false;
                app.pixels_done = 0;
            } else { app.ss = ss_prev; }
            app.need_redraw = true;
        }
        if gfx::key_pressed("r") && !video_mode {
            app.recording = !app.recording;
            if app.recording {
                recorded.clear();
                video_fit_pending = false;
                if app.vibe { ss_prev = app.ss; app.ss = 1; }
                app.vibe = true; // C starts vibe zoom for a smooth trajectory.
                write_stderr(str("Recording started\n"));
            } else {
                app.vibe = false;
                app.ss = ss_prev;
                if recorded.len == 0 {
                    write_stderr(str("No frames recorded\n"));
                } else {
                    video_mode = true;
                    video_index = 0;
                    video_path = timestamped_name(".raw");
                    write_file(video_path, []);
                    has_keyframe = false;
                    write_stderr(str("Rendering ", recorded.len, " frames\n"));
                }
            }
        }
        if gfx::key_pressed("Insert") { app.bailout *= 2.0; app.need_redraw = true; }
        if gfx::key_pressed("Delete") { app.bailout = max(app.bailout / 2.0, 2.0); app.need_redraw = true; }
        if gfx::key_pressed("Home") {
            app.auto_iter = false; app.auto_state = AS_OFF; app.probe_pending = false;
            app.iter_max = min(app.iter_max * 2, MAX_ITER_MAX);
            app.orbit_valid = false; app.need_redraw = true;
        }
        if gfx::key_pressed("End") {
            app.auto_iter = false; app.auto_state = AS_OFF; app.probe_pending = false;
            app.iter_max = max(app.iter_max / 2, 1);
            app.orbit_valid = false; app.need_redraw = true;
        }
        if gfx::key_pressed("PageUp") { app.iter_per_color *= 2; app.need_redraw = true; }
        if gfx::key_pressed("PageDown") { app.iter_per_color = max(app.iter_per_color / 2, 1); app.need_redraw = true; }
        if gfx::key_pressed("[") { app.omni_a = max(app.omni_a / 2.0, 1e-6); app.need_redraw = true; }
        if gfx::key_pressed("]") { app.omni_a = min(app.omni_a * 2.0, 1.0); app.need_redraw = true; }
        if gfx::key_pressed("1") { app.ss = 1; app.need_redraw = true; }
        if gfx::key_pressed("2") { app.ss = 2; app.need_redraw = true; }
        if gfx::key_pressed("4") { app.ss = 4; app.need_redraw = true; }
        if gfx::key_pressed("8") { app.ss = 8; app.need_redraw = true; }
        if gfx::key_pressed(".") { app.period_tol = app.period_tol * 10.0; app.need_redraw = true; }
        if gfx::key_pressed(",") { app.period_tol = max(app.period_tol / 10.0, 1e-61); app.need_redraw = true; }
        if gfx::key_pressed("s") {
            let path = timestamped_name(".sce");
            if save_scene(path, app) { write_stderr(str("Saved '", path, "'\n")); }
        }
        if gfx::mouse_down(gfx::MOUSE_LEFT) {
            if app.vibe { bake_view(app); }
            let d = gfx::mouse_delta();
            app.c_re -= d.x * app.view_width / (app.w as f64);
            app.c_im -= d.y * app.view_width / (app.w as f64);
            app.orbit_valid = false; app.need_redraw = true;
        }
        let wheel = gfx::mouse_wheel();
        if wheel != 0.0 {
            if app.vibe { bake_view(app); }
            let mp = gfx::mouse_pos();
            let ctrl = gfx::key_down("Left Ctrl") || gfx::key_down("Right Ctrl");
            let shift = gfx::key_down("Left Shift") || gfx::key_down("Right Shift");
            let ax = if ctrl { 0.0 } else { mp.x as f64 - app.w as f64 / 2.0 };
            let ay = if ctrl { 0.0 } else { mp.y as f64 - app.h as f64 / 2.0 };
            let factor = if shift { 1.1 } else { 1.4 };
            let zoom = if wheel > 0.0 { factor } else { 1.0 / factor };
            let wx = app.c_re + ax * app.view_width / (app.w as f64);
            let wy = app.c_im - ay * app.view_width / (app.w as f64);
            app.view_width /= zoom;
            app.c_re = wx - ax * app.view_width / (app.w as f64);
            app.c_im = wy + ay * app.view_width / (app.w as f64);
            app.orbit_valid = false; app.need_redraw = true;
        }

        } // !video_mode

        // ---- collect probe results (adaptive iterations) ----
        if app.probe_pending || video_fit_pending {
            var pmore = true;
            while pmore {
                let pres, pok = qpoll<ProbeResult>();
                if !pok { pmore = false; break; }
                if pres.fit {
                    if pres.gen != app.gen || !video_fit_pending { continue; }
                    if pres.total >= 128
                        && (pres.changed as f64) / (pres.total as f64) < 0.005
                        && video_fit_attempt < 2
                        && video_fit_n > 1 {
                        let vv = recorded[video_index];
                        var floor_n = vv.recorded_iter_max / 64;
                        if floor_n < vv.iter_per_color * 4 { floor_n = vv.iter_per_color * 4; }
                        if floor_n < 1 { floor_n = 1; }
                        let next = max(video_fit_n / 4, floor_n);
                        if next >= video_fit_n { video_fit_pending = false; continue; }
                        video_fit_n = next;
                        video_fit_attempt += 1;
                        fit_video_n(app, recorded[video_index], video_fit_n,
                                    app.gen, video_fit_attempt);
                    } else {
                        video_fit_pending = false;
                        let v = recorded[video_index];
                        app.c_re = v.c_re; app.c_im = v.c_im; app.view_width = v.width;
                        app.iter_max = video_fit_n;
                        app.iter_per_color = v.iter_per_color;
                        app.formula = v.formula; app.omni_a = v.omni_a; app.bailout = v.bailout;
                        app.lx = v.lx; app.ly = v.ly; app.lz = v.lz;
                        app.color_inside = if v.color_inside { 1 } else { 0 };
                        app.ss = 2;
                        app.disp_scale = 1.0; app.disp_pan_x = 0.0; app.disp_pan_y = 0.0;
                        app.orbit_valid = false; app.need_redraw = true;
                        video_dispatch_index = video_index;
                        video_pending = true; kf_pending = true;
                    }
                    continue;
                }
                if pres.gen != app.gen { app.probe_pending = false; continue; }
                app.probe_changed = pres.changed;
                app.probe_total = pres.total;
                app.probe_pending = false;
                auto_iter_update(app, app.last_render_ms);
            }
        }

        // ---- collect results without blocking ----
        if app.rendering {
            var more = true;
            while more {
                let res, ok = qpoll<PixResult>();
                if !ok { more = false; break; }
                if res.gen != app.gen { continue; }
                for i in res.indices.len {
                    let idx = res.indices[i];
                    if idx >= 0 && idx < app.image.len {
                        // Write straight into the visible image: finished
                        // pixels expand as a circle around the cursor while
                        // unfinished pixels still show the previous frame.
                        app.image[idx] = res.colors[i];
                        app.render_buf[idx] = res.colors[i];
                    }
                }
                app.pixels_done += res.indices.len;
            }
            if app.pixels_done >= app.pixels_total {
                app.rendering = false;
                if video_pending {
                    append_frame(video_path, app.image, frame_bytes);
                    if kf_pending {
                        keyframe_view = recorded[video_dispatch_index];
                        keyframe_image = copy(app.image);
                        keyframe_index = video_dispatch_index;
                        has_keyframe = true;
                        kf_pending = false;
                    }
                    video_index += 1;
                    video_pending = false;
                }
                let render_ms = (clock() - app.render_start) * 1000.0;
                app.last_render_ms = render_ms;
                let can_probe = app.auto_iter && !app.vibe
                    && app.probe_width > AUTO_DEEP_WIDTH && is_finite(app.probe_width);
                if can_probe {
                    qput(Job {
                        gen: app.gen, w: app.w, h: app.h, ss: 1,
                        center_re: app.c_re, center_im: app.c_im, width: app.view_width,
                        bailout: app.bailout, iter_max: app.iter_max, iter_per_color: app.iter_per_color,
                        formula: app.formula, omni_a: app.omni_a, lx: app.lx, ly: app.ly, lz: app.lz,
                        color_inside: app.color_inside, row_start: -2, row_count: 0, indices: [],
                        is_probe: true, probe_iter2: 0, fit_mode: false, fit_attempt: 0, probe_deadline: 200.0, probe_min_samples: 512,
                        mouse_cx: app.mouse_cx, mouse_cy: app.mouse_cy, vibe_render: false,
                        delta_c_re: app.delta_c_re, delta_c_im: app.delta_c_im, period_tol: app.period_tol,
                        palette_size: app.palette.size, palette_data: copy(app.palette.data),
                        orbit_c_re: 0.0, orbit_c_im: 0.0
                    });
                    app.probe_pending = true;
                } else {
                    auto_iter_update(app, render_ms);
                }
                if app.vibe { app.need_redraw = true; }
            }
        }

        // ---- vibe zoom: transform the last complete frame, never block ----
        if app.vibe {
            app.vibe_since_dispatch += 1;
            let mp = gfx::mouse_pos();
            let sx = mp.x as f64 - (app.w as f64) / 2.0;
            let sy = mp.y as f64 - (app.h as f64) / 2.0;
            let wx = sx / app.disp_scale + app.disp_pan_x;
            let wy = sy / app.disp_scale + app.disp_pan_y;
            // Do not magnify a stale frame past ~1.2x.  On a deep view a CPU
            // render takes seconds; without this the zoom turns the last
            // complete frame into a 4-second soap bubble.
            let inc = if app.rendering && app.disp_scale >= 1.2 { 1.0 } else { 1.02 };
            app.disp_scale *= inc;
            app.disp_pan_x = wx - sx / app.disp_scale;
            app.disp_pan_y = wy - sy / app.disp_scale;
        }

        if app.recording && recorded.len < 4096 {
            let vs = make_view(app);
            if recorded.len == 0 || !same_view(recorded[recorded.len - 1], vs) {
                recorded.push(vs);
            }
        }

        // ---- video render: reproject a keyframe or render the view ----
        // ---- vibe: stop early and restart from the centre ----
        // The list is sorted by distance from the cursor, so the first 50%
        // of results are the inner ~70% of the disk -- the region that
        // survives the ongoing zoom.  The outer rim is off-screen by the
        // time it would be drawn.
        if app.vibe && !video_mode && app.pixels_total > 0 {
            let drift = app.disp_scale;
            // Stop as soon as the inner ring around the cursor is freshly
            // rendered.  The pixel list is distance-sorted, so pixels_done
            // counts exactly those centre pixels first.
            if app.rendering {
                // Let the frame finish, then restart from the cursor at the
                // current view.  No drift abort: that killed renders before
                // a single fresh pixel arrived.
                if drift >= VIBE_HARD_DRIFT {
                    dispatch_render(app);
                }
            } else if drift >= VIBE_MAX_DRIFT {
                // Idle: a fresh frame is already on screen, start the next
                // render immediately before the zoom magnifies it again.
                dispatch_render(app);
            }
        }

        if video_mode && !video_pending && !video_fit_pending && !app.rendering && video_index < recorded.len {
            let v = recorded[video_index];
            if has_keyframe && can_reuse_keyframe(keyframe_view, v, app.w, app.h,
                                                  video_index, keyframe_index) {
                warp_keyframe(keyframe_view, v, app.image, keyframe_image, app.w, app.h);
                append_frame(video_path, app.image, frame_bytes);
                video_index += 1;
            } else {
                // fit_video_n: shrink iter_max while the N-vs-4N probe still
                // sees no visible change (same 3 attempts / 4x shrink as C).
                var min_n = v.recorded_iter_max / 64;
                if min_n < v.iter_per_color * 4 { min_n = v.iter_per_color * 4; }
                if min_n < 1 { min_n = 1; }
                video_fit_n = max(v.recorded_iter_max, min_n);
                video_fit_attempt = 0;
                video_fit_pending = true;
                fit_video_n(app, v, video_fit_n, app.gen, 0);
            }
        }
        if video_mode && !video_pending && video_index >= recorded.len {
            video_mode = false;
            write_stderr(str("Video saved to ", video_path, "\n"));
        }

        // No artificial 15-frame cooldown: in vibe the frame is stale the
        // moment it completes, so start the next one immediately.
        if app.need_redraw && !app.rendering && (!video_mode || video_pending) {
            dispatch_render(app);
        }

        // ---- draw the image through the display transform ----
        for i in app.w * app.h {
            let col = app.image[i];
            app.tex_bytes[i * 4] = ((col >> 16) & 0xFF) as u8;
            app.tex_bytes[i * 4 + 1] = ((col >> 8) & 0xFF) as u8;
            app.tex_bytes[i * 4 + 2] = (col & 0xFF) as u8;
            app.tex_bytes[i * 4 + 3] = ((col >> 24) & 0xFF) as u8;
        }
        gfx::update_texture(app.tex, app.tex_bytes);

        gfx::begin_screen(float4 { 0.0, 0.0, 0.0, 1.0 });
        gfx::bind(quad_pipeline);
        gfx::bind_texture(gfx::FRAGMENT, 0, app.tex, nearest);
        gfx::uniforms(gfx::FRAGMENT, 0, ViewP { p: float4 {
            app.disp_scale as f32,
            (app.disp_pan_x / (app.w as f64)) as f32,
            (app.disp_pan_y / (app.h as f64)) as f32,
            0.0 as f32 } });
        gfx::bind_vertex_buffer(qvb);
        gfx::draw(3);
        gfx::end_pass();

        if has_ui {
            ui::input_from_gfx(ctx);
            if app.show_info {
                if ui::begin(ctx, "Info", ui::rect(10, 10, 360, 420), ui::WINDOW_BORDER | ui::WINDOW_MOVABLE | ui::WINDOW_TITLE) {
                    ui::layout_row_dynamic(ctx, 18.0, 1);
                    var line: u8[>..] = [];
                    format(line, "center: ", app.c_re, " ", app.c_im); ui::label(ctx, line, ui::TEXT_LEFT);
                    line = []; format(line, "width: ", app.view_width); ui::label(ctx, line, ui::TEXT_LEFT);
                    line = []; format(line, "iter_max: ", app.iter_max); ui::label(ctx, line, ui::TEXT_LEFT);
                    line = []; format(line, "adaptive: ", auto_state_name(app.auto_state)); ui::label(ctx, line, ui::TEXT_LEFT);
                    line = []; format(line, "  sens: ", app.auto_sensitivity * 100.0, "% frame: ", app.auto_render_ms, "ms"); ui::label(ctx, line, ui::TEXT_LEFT);
                    line = []; format(line, "iter/color: ", app.iter_per_color); ui::label(ctx, line, ui::TEXT_LEFT);
                    line = []; format(line, "bailout: ", app.bailout); ui::label(ctx, line, ui::TEXT_LEFT);
                    line = []; format(line, "ssaa: ", app.ss); ui::label(ctx, line, ui::TEXT_LEFT);
                    line = []; format(line, "formula: ", app.formula); ui::label(ctx, line, ui::TEXT_LEFT);
                    line = []; format(line, "period_tol: ", app.period_tol); ui::label(ctx, line, ui::TEXT_LEFT);
                    line = []; format(line, "rendering: ", app.rendering); ui::label(ctx, line, ui::TEXT_LEFT);
                }
                ui::end(ctx);
            }
            if app.show_help {
                if ui::begin(ctx, "Help", ui::rect(40, 40, 420, 420), ui::WINDOW_BORDER | ui::WINDOW_MOVABLE | ui::WINDOW_TITLE) {
                    ui::layout_row_dynamic(ctx, 18.0, 1);
                    ui::label(ctx, "Alt info  F1/h help  v vibe  i auto", ui::TEXT_LEFT);
                    ui::label(ctx, "x reset  c center  b interior", ui::TEXT_LEFT);
                    ui::label(ctx, "m/o formula  [/] omni a (x2, /2)", ui::TEXT_LEFT);
                    ui::label(ctx, "Ins/Del bailout x2,/2  Home/End iter x2,/2", ui::TEXT_LEFT);
                    ui::label(ctx, "PgUp/PgDn iter/color x2,/2  1/2/4/8 SSAA", ui::TEXT_LEFT);
                    ui::label(ctx, "s save  r record  ,/. period tol", ui::TEXT_LEFT);
                    ui::label(ctx, "wheel zoom to cursor  Ctrl wheel center", ui::TEXT_LEFT);
                    ui::label(ctx, "Shift wheel fine zoom", ui::TEXT_LEFT);
                }
                ui::end(ctx);
            }
            ui::render(ctx, gfx::screen());
        }

        if gfx::key_pressed("F12") { gfx::screenshot("mandexplorer.png"); }
        drawn++;
        if frame_limit > 0 && drawn >= frame_limit { break; }
    }

    // Shut down workers.
    for i in nworkers {
        qput(Job {
            gen: 0, w: 0, h: 0, ss: 0,
            center_re: 0.0, center_im: 0.0, width: 0.0, bailout: 0.0,
            iter_max: 0, iter_per_color: 0, formula: 0, omni_a: 0.0,
            lx: 0.0, ly: 0.0, lz: 0.0, color_inside: 0,
            row_start: -1, row_count: 0, indices: [], is_probe: false, probe_iter2: 0, fit_mode: false, fit_attempt: 0, probe_deadline: 200.0, probe_min_samples: 512, mouse_cx: 0, mouse_cy: 0, vibe_render: false,
            delta_c_re: 0.0, delta_c_im: 0.0, period_tol: 0.0,
            palette_size: 0, palette_data: [],
            orbit_c_re: 0.0, orbit_c_im: 0.0
        });
    }
    for id in app.worker_ids { thread_wait(id); }

    if has_ui { ui::destroy(ctx); }
}
