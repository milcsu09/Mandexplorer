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
    mouse_cx: i64, mouse_cy: i64,
    delta_c_re: f64, delta_c_im: f64, period_tol: f64,
    palette_size: i64, palette_data: u32[],
    orbit_re: f64[], orbit_im: f64[], has_orbit: bool
}

// Flat result: a batch of pixels rendered in distance-from-mouse order.
// The main thread scatters them into the displayed image so the fresh
// pixels expand as a circle around the cursor (as in the original C build).
struct PixResult { gen: u64, indices: i32[], colors: u32[] }
struct Px { x: i64, y: i64, idx: i32, dx2: f64 }
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
    mouse_cx: i64, mouse_cy: i64,
    show_info: bool, show_help: bool,
    auto_iter: bool, auto_state: i32, auto_deep_zoom: bool, vibe: bool,
    period_tol: f64, recording: bool,
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

fn orbit_cache_reuse(app: AppState&) -> bool {
    if !app.orbit_valid { return false; }
    if app.formula == FORMULA_OMNI { return true; }
    if app.iter_max != app.orbit_iter_max { return false; }
    let dc_re = app.c_re - app.orbit_c_re;
    let dc_im = app.c_im - app.orbit_c_im;
    let mag2 = app.c_re * app.c_re + app.c_im * app.c_im;
    let tol = 1e-12;
    if dc_re * dc_re + dc_im * dc_im <= tol * tol * (1.0 + mag2) {
        app.delta_c_re = dc_re;
        app.delta_c_im = dc_im;
        return true;
    }
    return false;
}

// Perturbation iteration with period detection. Returns smooth iteration
// count nu and the derivative normal (nx, ny) used for shading.
fn mandel(app: AppState&, delta_re: f64, delta_im: f64) -> f64, f64, f64 {
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
        let r_re = app.orbit_re[io];
        let r_im = app.orbit_im[io];
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

        let r1_re = app.orbit_re[io + 1];
        let r1_im = app.orbit_im[io + 1];
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

fn pixel_color(app: AppState&, dx: f64, dy: f64) -> u32 {
    let nu, nx, ny = mandel(app, dx, dy);
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
fn render_sorted(job: Job&, y0: i64, count: i64) {
    var pa = AppState {
        c_re: job.center_re, c_im: job.center_im, view_width: job.width,
        bailout: job.bailout, lx: job.lx, ly: job.ly, lz: job.lz,
        iter_max: job.iter_max, iter_per_color: job.iter_per_color,
        formula: job.formula, omni_a: job.omni_a, color_inside: job.color_inside,
        disp_scale: 1.0, disp_pan_x: 0.0, disp_pan_y: 0.0,
        pend_re: 0.0, pend_im: 0.0, pend_width: job.width, pend_valid: false,
        snap_scale: 1.0, snap_pan_x: 0.0, snap_pan_y: 0.0,
        w: job.w, h: job.h, ss: job.ss, image: [], render_buf: [],
        orbit_re: copy(job.orbit_re), orbit_im: copy(job.orbit_im),
        orbit_valid: job.has_orbit, orbit_iter_max: job.iter_max,
        orbit_c_re: job.center_re, orbit_c_im: job.center_im,
        delta_c_re: job.delta_c_re, delta_c_im: job.delta_c_im,
        palette: Palette { size: job.palette_size, data: copy(job.palette_data) },
        palette_path: [],
        gen: job.gen, rendering: false, need_redraw: false,
        pixels_done: 0, pixels_total: 0, mouse_cx: job.mouse_cx, mouse_cy: job.mouse_cy,
        show_info: false, show_help: false,
        auto_iter: false, auto_state: 0, auto_deep_zoom: false, vibe: false,
        period_tol: job.period_tol, recording: false,
        tex: gfx::Texture { id: 0 }, worker_ids: []
    };

    // Build this worker's slice of pixels, sorted by distance from the mouse.
    var pts: Px[>..] = [];
    let cx = job.mouse_cx as f64;
    let cy = job.mouse_cy as f64;
    for ry in count {
        let y = y0 + ry;
        if y >= job.h { break; }
        for x in job.w {
            let dx = (x as f64) - cx;
            let dy = (y as f64) - cy;
            pts.push(Px { x: x, y: y, idx: ((y * job.w + x) as i32), dx2: dx * dx + dy * dy });
        }
    }
    sort(pts) { a, b => a.dx2 < b.dx2 };

    let ss = job.ss;
    var idxs: i32[>..] = [];
    var cols: u32[>..] = [];
    for p in pts {
        var r: u32 = 0; var g: u32 = 0; var b: u32 = 0;
        for sy in ss {
            for sx in ss {
                let fx = (p.x as f64) + (sx as f64 + 0.5) / (ss as f64);
                let fy = (p.y as f64) + (sy as f64 + 0.5) / (ss as f64);
                let dx = (fx - (job.w as f64) / 2.0) * job.width / (job.w as f64);
                let dy = ((job.h as f64) / 2.0 - fy) * job.width / (job.w as f64);
                let col = pixel_color(pa, dx, dy);
                r += (col >> 16) & 0xFF;
                g += (col >> 8) & 0xFF;
                b += col & 0xFF;
            }
        }
        let n = (ss * ss) as u32;
        idxs.push(p.idx);
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
    loop {
        let job = qget<Job>();
        if job.row_start < 0 { break; }
        render_sorted(job, job.row_start, job.row_count);
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
    // Enqueue row bands in order of distance from the mouse, so workers pick
    // up the nearest bands first and the visible fill expands as a circle.
    let rows_per_job = 8;
    var jobs: RowJob[>..] = [];
    var y = 0;
    while y < app.h {
        let cnt = if y + rows_per_job < app.h { rows_per_job } else { app.h - y };
        let dy = ((y as f64) + (cnt as f64) / 2.0) - (app.mouse_cy as f64);
        let dx = (app.w as f64) / 2.0 - (app.mouse_cx as f64);
        jobs.push(RowJob { y: y, cnt: cnt, r2: dx * dx + dy * dy });
        y = y + cnt;
    }
    sort(jobs) { a, b => a.r2 < b.r2 };
    for j in jobs {
        qput(Job {
            gen: app.gen, w: app.w, h: app.h, ss: app.ss,
            center_re: app.c_re, center_im: app.c_im, width: app.view_width,
            bailout: app.bailout, iter_max: app.iter_max, iter_per_color: app.iter_per_color,
            formula: app.formula, omni_a: app.omni_a, lx: app.lx, ly: app.ly, lz: app.lz,
            color_inside: app.color_inside, row_start: j.y, row_count: j.cnt,
            mouse_cx: app.mouse_cx, mouse_cy: app.mouse_cy,
            delta_c_re: app.delta_c_re, delta_c_im: app.delta_c_im, period_tol: app.period_tol,
            palette_size: app.palette.size, palette_data: copy(app.palette.data),
            orbit_re: copy(app.orbit_re), orbit_im: copy(app.orbit_im), has_orbit: app.orbit_valid
        });
    }
}

fn bake_display(app: AppState&) {
    if app.disp_scale == 1.0 && app.disp_pan_x == 0.0 && app.disp_pan_y == 0.0 { return; }
    var out: u32[>..] = [];
    out.resize(app.w * app.h, 0xFF000000 as u32);
    let sc = app.disp_scale;
    let px = app.disp_pan_x / (app.w as f64);
    let py = app.disp_pan_y / (app.h as f64);
    for sy in app.h {
        let vy = ((sy as f64 + 0.5) / (app.h as f64) - 0.5) / sc + 0.5 + py;
        let srcy = clamp((vy * (app.h as f64) + 0.5) as i64, 0, app.h - 1);
        for sx in app.w {
            let vx = ((sx as f64 + 0.5) / (app.w as f64) - 0.5) / sc + 0.5 + px;
            let srcx = clamp((vx * (app.w as f64) + 0.5) as i64, 0, app.w - 1);
            out[sy * app.w + sx] = app.image[srcy * app.w + srcx];
        }
    }
    app.image = copy(out);
    app.disp_scale = 1.0;
    app.disp_pan_x = 0.0;
    app.disp_pan_y = 0.0;
}

fn dispatch_render(app: AppState&) {
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
    compute_orbit(app);
    app.render_buf.resize(app.w * app.h, 0xFF000000 as u32);
    for i in app.w * app.h { app.render_buf[i] = app.image[i]; }
    enqueue_jobs(app);
    app.pixels_done = 0;
    app.pixels_total = app.w * app.h;
    app.rendering = true;
    app.need_redraw = false;
}

// Bake the accumulated vibe display transform into the committed scene and
// reset it, so the freshly completed frame and the display agree exactly.
fn commit_transform(app: AppState&) {
    app.pend_valid = false;
    let w = app.w as f64;
    app.c_re += app.disp_pan_x * app.view_width / w;
    app.c_im -= app.disp_pan_y * app.view_width / w;
    app.view_width /= app.disp_scale;
    app.disp_scale = 1.0;
    app.disp_pan_x = 0.0;
    app.disp_pan_y = 0.0;
    app.orbit_valid = false;
}

fn main() {
    let argv = args();
    var scene_path: u8[>..] = [];
    var frame_limit = -1;
    var ai = 1;
    while ai < argv.len {
        if argv[ai] == "--frames" && ai + 1 < argv.len {
            let n, ok = parse_int(argv[ai + 1]);
            if ok { frame_limit = n; }
            ai += 2;
        } else {
            scene_path.append(argv[ai]);
            ai += 1;
        }
    }
    if scene_path.len == 0 { scene_path.append("default.sce"); }

    var app = AppState {
        c_re: 0.0, c_im: 0.0, view_width: 8.0,
        bailout: 2.0, lx: 0.577, ly: 0.577, lz: 0.577,
        iter_max: 1024, iter_per_color: 16,
        formula: FORMULA_MANDEL, omni_a: 0.1, color_inside: 0,
        disp_scale: 1.0, disp_pan_x: 0.0, disp_pan_y: 0.0,
        pend_re: 0.0, pend_im: 0.0, pend_width: 8.0, pend_valid: false,
        snap_scale: 1.0, snap_pan_x: 0.0, snap_pan_y: 0.0,
        w: 800, h: 600, ss: 1, image: [], render_buf: [],
        orbit_re: [], orbit_im: [], orbit_valid: false,
        orbit_iter_max: 0, orbit_c_re: 0.0, orbit_c_im: 0.0,
        delta_c_re: 0.0, delta_c_im: 0.0,
        palette: Palette { size: 0, data: [] }, palette_path: [],
        gen: 0, rendering: false, need_redraw: false,
        pixels_done: 0, pixels_total: 0, mouse_cx: 400, mouse_cy: 300,
        show_info: true, show_help: false,
        auto_iter: false, auto_state: 0, auto_deep_zoom: false, vibe: false,
        period_tol: 1e-40, recording: false,
        tex: gfx::Texture { id: 0 }, worker_ids: []
    };

    if !load_scene(scene_path, app) {
        write_stderr(str("Failed to load scene\n"));
        return;
    }
    var pal_path: u8[>..] = [];
    if app.palette_path.len > 0 { pal_path.append(app.palette_path); }
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
    if app.tex.id == 0 {
        write_stderr(str("texture fail\n"));
        return;
    }
    let nearest = gfx::sampler(gfx::NEAREST, gfx::CLAMP);

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
    var recording_file: u8[>..] = [];
    recording_file.append("recording.raw");

    while gfx::frame() {
        let sz = gfx::screen_size();
        if sz.x as i64 != app.w || sz.y as i64 != app.h {
            app.w = sz.x as i64;
            app.h = sz.y as i64;
            app.image.resize(app.w * app.h, 0xFF000000 as u32);
            app.render_buf.resize(app.w * app.h, 0);
            app.tex = gfx::texture2d(app.w, app.h);
            app.need_redraw = true;
        }

        // ---- input ----
        if gfx::key_pressed("Escape") { break; }
        if gfx::key_pressed("Left Alt") { app.show_info = !app.show_info; }
        if gfx::key_pressed("F1") { app.show_help = !app.show_help; }
        if gfx::key_pressed("h") { app.show_help = !app.show_help; }
        if gfx::key_pressed("x") {
            if app.vibe { commit_transform(app); }
            app.c_re = 0.0; app.c_im = 0.0; app.view_width = 8.0; app.iter_max = 1024;
            app.orbit_valid = false; app.need_redraw = true;
        }
        if gfx::key_pressed("c") {
            if app.vibe { commit_transform(app); }
            let mp = gfx::mouse_pos();
            let dx = (mp.x as f64 - app.w as f64 / 2.0) * app.view_width / (app.w as f64);
            let dy = (mp.y as f64 - app.h as f64 / 2.0) * app.view_width / (app.w as f64);
            app.c_re += dx; app.c_im += dy;
            app.orbit_valid = false; app.need_redraw = true;
        }
        if gfx::key_pressed("b") { app.color_inside = if app.color_inside == 1 { 0 } else { 1 }; app.need_redraw = true; }
        if gfx::key_pressed("m") { app.formula = FORMULA_MANDEL; app.orbit_valid = false; app.need_redraw = true; }
        if gfx::key_pressed("o") { app.formula = FORMULA_OMNI; app.orbit_valid = false; app.need_redraw = true; }
        if gfx::key_pressed("i") { app.auto_iter = !app.auto_iter; }
        if gfx::key_pressed("v") {
            // Do NOT bake/reset the display transform here.  `image` still
            // holds the last complete frame in the pre-transform scene, so
            // resetting disp to identity would snap the screen back to that
            // old frame until a fresh render finishes (the "zig-zag").
            // Freezing disp keeps the current view pixel-identical; the
            // completion rebase folds it into the next committed scene.
            app.vibe = !app.vibe;
            if app.vibe { ss_prev = app.ss; app.ss = 1; }
            else { app.ss = ss_prev; }
            app.need_redraw = true;
        }
        if gfx::key_pressed("r") {
            app.recording = !app.recording;
            if app.recording {
                var empty: u8[>..] = [];
                write_file(recording_file, empty);
            } else {
                write_stderr(str("Recording saved to recording.raw\n"));
            }
        }
        if gfx::key_pressed("Insert") { app.bailout += 1.0; app.need_redraw = true; }
        if gfx::key_pressed("Delete") { if app.bailout > 1.0 { app.bailout -= 1.0; } app.need_redraw = true; }
        if gfx::key_pressed("Home") { app.iter_max += 256; app.orbit_valid = false; app.need_redraw = true; }
        if gfx::key_pressed("End") { if app.iter_max > 256 { app.iter_max -= 256; } app.orbit_valid = false; app.need_redraw = true; }
        if gfx::key_pressed("PageUp") { app.iter_per_color += 1; app.need_redraw = true; }
        if gfx::key_pressed("PageDown") { if app.iter_per_color > 1 { app.iter_per_color -= 1; } app.need_redraw = true; }
        if gfx::key_pressed("[") { app.omni_a -= 0.01; app.need_redraw = true; }
        if gfx::key_pressed("]") { app.omni_a += 0.01; app.need_redraw = true; }
        if gfx::key_pressed("1") { app.ss = 1; app.need_redraw = true; }
        if gfx::key_pressed("2") { app.ss = 2; app.need_redraw = true; }
        if gfx::key_pressed("4") { app.ss = 4; app.need_redraw = true; }
        if gfx::key_pressed("8") { app.ss = 8; app.need_redraw = true; }
        if gfx::key_pressed(".") { app.period_tol = app.period_tol * 10.0; if app.period_tol > 1.0 { app.period_tol = 1.0; } app.need_redraw = true; }
        if gfx::key_pressed(",") { app.period_tol = app.period_tol / 10.0; if app.period_tol < 1e-61 { app.period_tol = 1e-61; } app.need_redraw = true; }
        if gfx::key_pressed("s") { save_scene("saved.sce", app); }
        if gfx::mouse_down(gfx::MOUSE_LEFT) {
            if app.vibe { commit_transform(app); }
            let d = gfx::mouse_delta();
            app.c_re -= d.x * app.view_width / (app.w as f64);
            app.c_im -= d.y * app.view_width / (app.w as f64);
            app.orbit_valid = false; app.need_redraw = true;
        }
        let wheel = gfx::mouse_wheel();
        if wheel != 0.0 {
            if app.vibe { commit_transform(app); }
            app.view_width *= 1.0 - (wheel as f64) * 0.1;
            if app.view_width < 1e-15 { app.view_width = 1e-15; }
            app.orbit_valid = false; app.need_redraw = true;
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
                if app.vibe { app.need_redraw = true; }
            }
        }

        // ---- vibe zoom: transform the last complete frame, never block ----
        if app.vibe {
            let mp = gfx::mouse_pos();
            let sx = mp.x as f64 - (app.w as f64) / 2.0;
            let sy = mp.y as f64 - (app.h as f64) / 2.0;
            let wx = sx / app.disp_scale + app.disp_pan_x;
            let wy = sy / app.disp_scale + app.disp_pan_y;
            app.disp_scale *= 1.02;
            app.disp_pan_x = wx - sx / app.disp_scale;
            app.disp_pan_y = wy - sy / app.disp_scale;
        }

        if app.need_redraw && !app.rendering { dispatch_render(app); }

        // ---- recording ----
        if app.recording {
            let px = gfx::read_pixels(gfx::screen());
            var rgb: u8[>..] = [];
            for i in app.w * app.h {
                rgb.push(px[i * 4]); rgb.push(px[i * 4 + 1]); rgb.push(px[i * 4 + 2]);
            }
            var existing: u8[>..] = [];
            read_file(recording_file, existing);
            existing.append(rgb);
            write_file(recording_file, existing);
        }

        // ---- draw the image through the display transform ----
        var bytes: u8[>..] = [];
        bytes.resize(app.w * app.h * 4, 0);
        for i in app.w * app.h {
            let col = app.image[i];
            bytes[i * 4] = ((col >> 16) & 0xFF) as u8;
            bytes[i * 4 + 1] = ((col >> 8) & 0xFF) as u8;
            bytes[i * 4 + 2] = (col & 0xFF) as u8;
            bytes[i * 4 + 3] = ((col >> 24) & 0xFF) as u8;
        }
        gfx::update_texture(app.tex, bytes);

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
                    ui::label(ctx, "m/o formula  [/] omni a", ui::TEXT_LEFT);
                    ui::label(ctx, "Insert/Delete bailout  Home/End iter", ui::TEXT_LEFT);
                    ui::label(ctx, "PgUp/PgDn iter/color  1/2/4/8 SSAA", ui::TEXT_LEFT);
                    ui::label(ctx, "s save  r record  ,/. period tol", ui::TEXT_LEFT);
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
            row_start: -1, row_count: 0, mouse_cx: 0, mouse_cy: 0,
            delta_c_re: 0.0, delta_c_im: 0.0, period_tol: 0.0,
            palette_size: 0, palette_data: [],
            orbit_re: [], orbit_im: [], has_orbit: false
        });
    }
    for id in app.worker_ids { thread_wait(id); }

    if has_ui { ui::destroy(ctx); }
}
