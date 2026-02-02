"""
Plate Bending Analysis - Triple Method Comparison
==================================================
1. Lévy Method (numerically stable exponential formulation)
2. Finite Integral Transform Method (with auto-convergence)
3. Rayleigh-Ritz Method (independent validation)

Features async solving with progress indicators.
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import threading
import queue
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import from package
from plate_bending.solvers.levy_solver import StableLevySolver
from plate_bending.solvers.fit_solver import FITSolver
from plate_bending.solvers.ritz_solver import RitzSolver
from plate_bending.validation.benchmarks import Benchmarks


class TripleMethodGUI:
    """
    GUI with three solver methods running asynchronously:
    1. Levy (primary, fast, exact for Levy-type BCs)
    2. FIT (with auto-convergence and Lanczos smoothing)
    3. Ritz (general, independent validation)
    """

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Plate Bending - Triple Method Comparison")
        self.root.geometry("1600x1000")

        self.levy_results = None
        self.fit_results = None
        self.ritz_results = None
        self.fit_valid = True  # Default to valid until checked

        # Queue for thread communication
        self.result_queue = queue.Queue()

        # Solver status
        self.solver_running = {'levy': False, 'fit': False, 'ritz': False}

        self._build_gui()

        # Start polling for results
        self._poll_results()

    def _build_gui(self):
        main = ttk.Frame(self.root, padding="5")
        main.pack(fill=tk.BOTH, expand=True)

        # Left panel
        left = ttk.Frame(main, width=400)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0,5))
        left.pack_propagate(False)

        # Geometry
        gf = ttk.LabelFrame(left, text="Geometry", padding="5")
        gf.pack(fill=tk.X, pady=3, padx=5)
        for i, (lbl, val) in enumerate([("a (m):", "1.0"), ("b (m):", "1.0"), ("h (m):", "0.01")]):
            ttk.Label(gf, text=lbl).grid(row=i, column=0, sticky=tk.W)
            e = ttk.Entry(gf, width=10)
            e.insert(0, val)
            e.grid(row=i, column=1)
            setattr(self, f"e_{['a','b','h'][i]}", e)

        # Material
        mf = ttk.LabelFrame(left, text="Material", padding="5")
        mf.pack(fill=tk.X, pady=3, padx=5)
        ttk.Label(mf, text="E (Pa):").grid(row=0, column=0)
        self.e_E = ttk.Entry(mf, width=10)
        self.e_E.insert(0, "2.1e11")
        self.e_E.grid(row=0, column=1)
        ttk.Label(mf, text="v:").grid(row=1, column=0)
        self.e_nu = ttk.Entry(mf, width=10)
        self.e_nu.insert(0, "0.3")
        self.e_nu.grid(row=1, column=1)

        # BCs
        bcf = ttk.LabelFrame(left, text="Boundary Conditions", padding="5")
        bcf.pack(fill=tk.X, pady=3, padx=5)

        ttk.Label(bcf, text="Levy: x=0 & x=a must be S (Simply Supported)",
                 font=('TkDefaultFont', 8), foreground='blue').pack()

        bc_inner = ttk.Frame(bcf)
        bc_inner.pack(fill=tk.X, pady=3)

        ttk.Label(bc_inner, text="y=0:").grid(row=0, column=0)
        self.bc_y0 = tk.StringVar(value='C')
        ttk.Combobox(bc_inner, textvariable=self.bc_y0, values=['S','C','F'], width=4).grid(row=0, column=1)

        ttk.Label(bc_inner, text="y=b:").grid(row=0, column=2, padx=(10,0))
        self.bc_yb = tk.StringVar(value='F')
        ttk.Combobox(bc_inner, textvariable=self.bc_yb, values=['S','C','F'], width=4).grid(row=0, column=3)

        preset_f = ttk.Frame(bcf)
        preset_f.pack(fill=tk.X, pady=3)
        for i, (name, y0, yb) in enumerate([('SSSS','S','S'), ('SCSC','C','C'), ('SCSS','C','S'),
                                            ('SCSF','C','F'), ('SSSF','S','F'), ('SFSF','F','F')]):
            ttk.Button(preset_f, text=name, width=6,
                      command=lambda a=y0,b=yb: (self.bc_y0.set(a), self.bc_yb.set(b))).grid(row=i//3, column=i%3, padx=1, pady=1)

        # Loading
        lf = ttk.LabelFrame(left, text="Loading", padding="5")
        lf.pack(fill=tk.X, pady=3, padx=5)

        self.load_var = tk.StringVar(value='uniform')
        for txt, val in [("Uniform q0", "uniform"), ("Rect Patch", "rect_patch"),
                         ("Circular Patch", "circular"), ("Point P", "point")]:
            ttk.Radiobutton(lf, text=txt, variable=self.load_var, value=val,
                           command=self._toggle_load).pack(anchor=tk.W)

        ttk.Label(lf, text="q0 (Pa):").pack(anchor=tk.W)
        self.e_q0 = ttk.Entry(lf, width=10)
        self.e_q0.insert(0, "10000")
        self.e_q0.pack(anchor=tk.W)

        # Point/circular position
        self.pos_frame = ttk.Frame(lf)
        ttk.Label(self.pos_frame, text="x0/a:").grid(row=0, column=0)
        self.e_x0 = ttk.Entry(self.pos_frame, width=5)
        self.e_x0.insert(0, "0.5")
        self.e_x0.grid(row=0, column=1)
        ttk.Label(self.pos_frame, text="y0/b:").grid(row=0, column=2)
        self.e_y0 = ttk.Entry(self.pos_frame, width=5)
        self.e_y0.insert(0, "0.5")
        self.e_y0.grid(row=0, column=3)

        # Circular radius
        self.R_frame = ttk.Frame(lf)
        ttk.Label(self.R_frame, text="R/a:").grid(row=0, column=0)
        self.e_R = ttk.Entry(self.R_frame, width=5)
        self.e_R.insert(0, "0.1")
        self.e_R.grid(row=0, column=1)

        # Patch bounds
        self.patch_frame = ttk.Frame(lf)
        ttk.Label(self.patch_frame, text="x1/a:").grid(row=0, column=0)
        self.e_x1 = ttk.Entry(self.patch_frame, width=5)
        self.e_x1.insert(0, "0.3")
        self.e_x1.grid(row=0, column=1)
        ttk.Label(self.patch_frame, text="x2/a:").grid(row=0, column=2)
        self.e_x2 = ttk.Entry(self.patch_frame, width=5)
        self.e_x2.insert(0, "0.7")
        self.e_x2.grid(row=0, column=3)
        ttk.Label(self.patch_frame, text="y1/b:").grid(row=1, column=0)
        self.e_y1 = ttk.Entry(self.patch_frame, width=5)
        self.e_y1.insert(0, "0.3")
        self.e_y1.grid(row=1, column=1)
        ttk.Label(self.patch_frame, text="y2/b:").grid(row=1, column=2)
        self.e_y2 = ttk.Entry(self.patch_frame, width=5)
        self.e_y2.insert(0, "0.7")
        self.e_y2.grid(row=1, column=3)

        # Solver options
        sf = ttk.LabelFrame(left, text="Solver Options", padding="5")
        sf.pack(fill=tk.X, pady=3, padx=5)
        ttk.Label(sf, text="n (Levy/FIT):").grid(row=0, column=0)
        self.e_n = ttk.Entry(sf, width=6)
        self.e_n.insert(0, "50")
        self.e_n.grid(row=0, column=1)

        self.ritz_enabled = tk.BooleanVar(value=True)
        ttk.Checkbutton(sf, text="Ritz", variable=self.ritz_enabled).grid(row=1, column=0, sticky=tk.W)
        ttk.Label(sf, text="M:").grid(row=1, column=1)
        self.e_ritz_M = ttk.Entry(sf, width=4)
        self.e_ritz_M.insert(0, "15")
        self.e_ritz_M.grid(row=1, column=2)
        ttk.Label(sf, text="N:").grid(row=1, column=3)
        self.e_ritz_N = ttk.Entry(sf, width=4)
        self.e_ritz_N.insert(0, "15")
        self.e_ritz_N.grid(row=1, column=4)

        self.fit_autoconv = tk.BooleanVar(value=True)
        ttk.Checkbutton(sf, text="FIT Auto-Converge", variable=self.fit_autoconv).grid(row=2, column=0, columnspan=3, sticky=tk.W)

        # Note: FIT now supports all Levy-type BCs (SS on x-edges) via extended method
        # The skip option is kept for cases where user wants Levy-only comparison
        self.fit_skip_invalid = tk.BooleanVar(value=False)  # Default to enabled since FIT now works
        ttk.Checkbutton(sf, text="Skip FIT solver", variable=self.fit_skip_invalid).grid(row=3, column=0, columnspan=5, sticky=tk.W)

        # Run button
        ttk.Button(left, text="Run All Solvers", command=self._run_all,
                  style='Accent.TButton').pack(pady=10, padx=5, fill=tk.X)

        # Progress indicators
        pf = ttk.LabelFrame(left, text="Solver Progress", padding="5")
        pf.pack(fill=tk.X, pady=3, padx=5)

        # Levy progress
        levy_f = ttk.Frame(pf)
        levy_f.pack(fill=tk.X, pady=2)
        self.levy_status = ttk.Label(levy_f, text="Levy: Ready", width=20, anchor=tk.W)
        self.levy_status.pack(side=tk.LEFT)
        self.levy_progress = ttk.Progressbar(levy_f, length=150, mode='indeterminate')
        self.levy_progress.pack(side=tk.LEFT, padx=5)

        # FIT progress
        fit_f = ttk.Frame(pf)
        fit_f.pack(fill=tk.X, pady=2)
        self.fit_status = ttk.Label(fit_f, text="FIT: Ready", width=20, anchor=tk.W)
        self.fit_status.pack(side=tk.LEFT)
        self.fit_progress = ttk.Progressbar(fit_f, length=150, mode='determinate')
        self.fit_progress.pack(side=tk.LEFT, padx=5)

        # Ritz progress
        ritz_f = ttk.Frame(pf)
        ritz_f.pack(fill=tk.X, pady=2)
        self.ritz_status = ttk.Label(ritz_f, text="Ritz: Ready", width=20, anchor=tk.W)
        self.ritz_status.pack(side=tk.LEFT)
        self.ritz_progress = ttk.Progressbar(ritz_f, length=150, mode='indeterminate')
        self.ritz_progress.pack(side=tk.LEFT, padx=5)

        # Plot options
        pof = ttk.LabelFrame(left, text="Plot Options", padding="5")
        pof.pack(fill=tk.X, pady=3, padx=5)
        self.plot_var = tk.StringVar(value='W')
        for txt, val in [("Deflection W", "W"), ("Moment Mx", "Mx"),
                         ("Von Mises", "vm"), ("3D Surface", "3D")]:
            ttk.Radiobutton(pof, text=txt, variable=self.plot_var, value=val,
                           command=self._update_plot).pack(anchor=tk.W)

        # Right panel - notebook
        right = ttk.Frame(main)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.nb = ttk.Notebook(right)
        self.nb.pack(fill=tk.BOTH, expand=True)

        # Comparison tab
        comp_tab = ttk.Frame(self.nb)
        self.nb.add(comp_tab, text="Comparison")

        self.comp_text = scrolledtext.ScrolledText(comp_tab, height=12, font=('Courier', 9))
        self.comp_text.pack(fill=tk.X, padx=5, pady=5)

        pf = ttk.Frame(comp_tab)
        pf.pack(fill=tk.BOTH, expand=True)
        self.fig = plt.figure(figsize=(14, 5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=pf)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(self.canvas, pf)

        # Debug tab
        debug_tab = ttk.Frame(self.nb)
        self.nb.add(debug_tab, text="Debug")
        self.debug_text = scrolledtext.ScrolledText(debug_tab, font=('Courier', 9))
        self.debug_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def _toggle_load(self):
        load = self.load_var.get()
        self.pos_frame.pack_forget()
        self.R_frame.pack_forget()
        self.patch_frame.pack_forget()

        if load == 'uniform':
            pass
        elif load == 'point':
            self.pos_frame.pack(fill=tk.X, pady=2)
        elif load == 'circular':
            self.pos_frame.pack(fill=tk.X, pady=2)
            self.R_frame.pack(fill=tk.X, pady=2)
        elif load == 'rect_patch':
            self.patch_frame.pack(fill=tk.X, pady=2)

    def _get_params(self):
        """Get all input parameters."""
        a = float(self.e_a.get())
        b = float(self.e_b.get())
        h = float(self.e_h.get())
        E = float(self.e_E.get())
        nu = float(self.e_nu.get())
        q0 = float(self.e_q0.get())
        n = int(self.e_n.get())
        bc_y0 = self.bc_y0.get()
        bc_yb = self.bc_yb.get()
        bc = f"S{bc_y0}S{bc_yb}"
        load = self.load_var.get()

        x0 = float(self.e_x0.get()) * a if load in ['point', 'circular'] else None
        y0 = float(self.e_y0.get()) * b if load in ['point', 'circular'] else None
        R = float(self.e_R.get()) * a if load == 'circular' else None
        x1 = float(self.e_x1.get()) * a if load == 'rect_patch' else None
        y1 = float(self.e_y1.get()) * b if load == 'rect_patch' else None
        x2 = float(self.e_x2.get()) * a if load == 'rect_patch' else None
        y2 = float(self.e_y2.get()) * b if load == 'rect_patch' else None

        M = int(self.e_ritz_M.get())
        N = int(self.e_ritz_N.get())

        return {
            'a': a, 'b': b, 'h': h, 'E': E, 'nu': nu, 'q0': q0, 'n': n,
            'bc_y0': bc_y0, 'bc_yb': bc_yb, 'bc': bc, 'load': load,
            'x0': x0, 'y0': y0, 'R': R, 'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
            'M': M, 'N': N
        }

    def _run_all(self):
        """Run all solvers asynchronously."""
        try:
            params = self._get_params()
        except Exception as e:
            messagebox.showerror("Input Error", str(e))
            return

        # Reset results
        self.levy_results = None
        self.fit_results = None
        self.ritz_results = None

        # Clear comparison
        self.comp_text.delete(1.0, tk.END)
        self.comp_text.insert(tk.END, "Running solvers...\n")

        # Check if FIT is valid for this BC
        # FIT works for all Levy-type plates (SS on x-edges: S___S_)
        # Format: [x=0][y=0][x=a][y=b]
        bc = params['bc']
        self.fit_valid = (bc[0] == 'S' and bc[2] == 'S')  # Levy-type BCs

        # Start Levy solver
        self._start_levy(params)

        # Start FIT solver
        if self.fit_skip_invalid.get():
            self.fit_status.config(text="FIT: Skipped")
            self.fit_results = None
        elif self.fit_valid:
            self._start_fit(params)
        else:
            # Non-Levy BCs not currently supported by FIT
            self.fit_status.config(text="FIT: N/A (non-Levy BC)")
            self.fit_results = None

        # Start Ritz solver
        if self.ritz_enabled.get():
            self._start_ritz(params)
        else:
            self.ritz_status.config(text="Ritz: Disabled")

    def _start_levy(self, params):
        """Start Levy solver in background thread."""
        self.solver_running['levy'] = True
        self.levy_status.config(text="Levy: Running...")
        self.levy_progress.start(10)

        def run():
            try:
                solver = StableLevySolver(
                    params['a'], params['b'], params['h'],
                    params['E'], params['nu'],
                    params['bc_y0'], params['bc_yb'], params['n']
                )
                result = solver.solve(
                    params['load'], params['q0'],
                    params['x0'], params['y0'], params['R'],
                    params['x1'], params['y1'], params['x2'], params['y2']
                )
                self.result_queue.put(('levy', result, solver.debug))
            except Exception as e:
                self.result_queue.put(('levy_error', str(e), []))

        threading.Thread(target=run, daemon=True).start()

    def _start_fit(self, params):
        """Start FIT solver in background thread."""
        self.solver_running['fit'] = True
        self.fit_status.config(text="FIT: Running...")
        self.fit_progress['value'] = 0

        def progress_callback(progress, message):
            self.result_queue.put(('fit_progress', progress, message))

        def run():
            try:
                solver = FITSolver(
                    params['a'], params['b'], params['h'],
                    params['E'], params['nu'],
                    params['bc'], params['n']
                )
                result = solver.solve(
                    params['load'], params['q0'],
                    params['x0'], params['y0'], params['R'],
                    params['x1'], params['y1'], params['x2'], params['y2'],
                    auto_converge=self.fit_autoconv.get(),
                    progress_callback=progress_callback
                )
                self.result_queue.put(('fit', result, solver.debug))
            except Exception as e:
                self.result_queue.put(('fit_error', str(e), []))

        threading.Thread(target=run, daemon=True).start()

    def _start_ritz(self, params):
        """Start Ritz solver in background thread."""
        self.solver_running['ritz'] = True
        self.ritz_status.config(text="Ritz: Running...")
        self.ritz_progress.start(10)

        def run():
            try:
                solver = RitzSolver(
                    params['a'], params['b'], params['h'],
                    params['E'], params['nu'],
                    params['bc'], params['M'], params['N']
                )
                result = solver.solve(
                    params['load'], params['q0'],
                    params['x0'], params['y0'],
                    params['x1'], params['y1'], params['x2'], params['y2'],
                    params['R']
                )
                self.result_queue.put(('ritz', result, solver.debug))
            except Exception as e:
                self.result_queue.put(('ritz_error', str(e), []))

        threading.Thread(target=run, daemon=True).start()

    def _poll_results(self):
        """Poll result queue and update GUI."""
        try:
            while True:
                msg = self.result_queue.get_nowait()
                self._handle_result(msg)
        except queue.Empty:
            pass

        # Continue polling
        self.root.after(100, self._poll_results)

    def _handle_result(self, msg):
        """Handle a result from the queue."""
        msg_type = msg[0]

        if msg_type == 'levy':
            self.levy_results = msg[1]
            self.levy_debug = msg[2]
            self.solver_running['levy'] = False
            self.levy_progress.stop()
            self.levy_status.config(text="Levy: Done")
            self._check_all_done()

        elif msg_type == 'levy_error':
            self.solver_running['levy'] = False
            self.levy_progress.stop()
            self.levy_status.config(text=f"Levy: Error")
            self.comp_text.insert(tk.END, f"Levy error: {msg[1]}\n")

        elif msg_type == 'fit':
            self.fit_results = msg[1]
            self.fit_debug = msg[2]
            self.solver_running['fit'] = False
            self.fit_progress['value'] = 100
            n_used = self.fit_results.get('n_terms_used', '?')
            method = self.fit_results.get('method', 'FIT')
            # Show method type (FIT or FIT-Extended)
            self.fit_status.config(text=f"{method}: Done (n={n_used})")
            self._check_all_done()

        elif msg_type == 'fit_progress':
            progress = msg[1]
            message = msg[2]
            self.fit_progress['value'] = progress * 100
            self.fit_status.config(text=message[:20])

        elif msg_type == 'fit_error':
            self.solver_running['fit'] = False
            self.fit_progress['value'] = 0
            self.fit_status.config(text=f"FIT: Error")
            self.comp_text.insert(tk.END, f"FIT error: {msg[1]}\n")

        elif msg_type == 'ritz':
            self.ritz_results = msg[1]
            self.ritz_debug = msg[2]
            self.solver_running['ritz'] = False
            self.ritz_progress.stop()
            self.ritz_status.config(text="Ritz: Done")
            self._check_all_done()

        elif msg_type == 'ritz_error':
            self.solver_running['ritz'] = False
            self.ritz_progress.stop()
            self.ritz_status.config(text=f"Ritz: Error")
            self.comp_text.insert(tk.END, f"Ritz error: {msg[1]}\n")

    def _check_all_done(self):
        """Check if all solvers are done and update display."""
        all_done = not any(self.solver_running.values())

        # Update comparison whenever we have new results
        self._update_comparison()
        self._update_plot()

        if all_done:
            self._update_debug()

    def _update_comparison(self):
        """Update the comparison text."""
        self.comp_text.delete(1.0, tk.END)

        L = self.levy_results
        F = self.fit_results
        R = self.ritz_results

        bc = f"S{self.bc_y0.get()}S{self.bc_yb.get()}"
        bench = Benchmarks.get(bc)

        lines = []
        lines.append("=" * 100)
        lines.append(f"  TRIPLE METHOD COMPARISON: {bc}")
        lines.append("=" * 100)

        # Warning for non-SSSS FIT
        if bc != 'SSSS':
            lines.append("")
            lines.append("  *** WARNING: FIT uses sine series (W=0 at edges) - INVALID for free/clamped edges! ***")
            lines.append("  *** For non-SSSS plates, use LEVY (primary) or RITZ (validation) results only.    ***")
            lines.append("")

        # Header
        fit_label = "FIT" if bc == 'SSSS' else "FIT*"
        header = f"{'':30} | {'LEVY':^17} | {fit_label:^17} |"
        if R:
            header += f" {'RITZ':^17} |"
        header += " BENCHMARK"
        lines.append(header)
        lines.append("-" * 100)

        # Data rows
        def fmt_val(val):
            if val is None:
                return "     ---     "
            return f"{val:15.6f}"

        def fmt_exp(val):
            if val is None:
                return "     ---     "
            return f"{val:15.4e}"

        # W_max
        row = f"  {'Max |W| (mm)':<28} |"
        row += f"  {fmt_val(L['W_max']*1e3 if L else None)}  |"
        row += f"  {fmt_val(F['W_max']*1e3 if F else None)}  |"
        if R:
            row += f"  {fmt_val(R['W_max']*1e3)}  |"
        lines.append(row)

        # W_coef
        row = f"  {'W coefficient (W*D/qa^4)':<28} |"
        row += f"  {fmt_val(L['W_coef'] if L else None)}  |"
        row += f"  {fmt_val(F['W_coef'] if F else None)}  |"
        if R:
            row += f"  {fmt_val(R['W_coef'])}  |"
        if bench:
            bv = bench.get('W_max_coef', bench.get('W_center_coef', 'N/A'))
            row += f" {bv}"
        lines.append(row)

        # Mx_max
        row = f"  {'Max |Mx| (N*m/m)':<28} |"
        row += f"  {fmt_exp(L['Mx_max'] if L else None)}  |"
        row += f"  {fmt_exp(F['Mx_max'] if F else None)}  |"
        if R:
            row += f"  {fmt_exp(R['Mx_max'])}  |"
        lines.append(row)

        # Von Mises
        row = f"  {'Max Von Mises (MPa)':<28} |"
        row += f"  {fmt_val(L['vm_max']/1e6 if L else None)}  |"
        row += f"  {fmt_val(F['vm_max']/1e6 if F else None)}  |"
        if R:
            row += f"  {fmt_val(R['vm_max']/1e6)}  |"
        lines.append(row)

        lines.append("-" * 100)

        # Differences
        if L and R:
            diff_lr = abs(L['W_coef'] - R['W_coef']) / L['W_coef'] * 100
            lines.append(f"  Levy vs Ritz difference: {diff_lr:.2f}%")
        if L and F:
            diff_lf = abs(L['W_coef'] - F['W_coef']) / L['W_coef'] * 100
            lines.append(f"  Levy vs FIT difference:  {diff_lf:.2f}%")

        lines.append("=" * 100)

        self.comp_text.insert(tk.END, "\n".join(lines))

    def _update_plot(self):
        """Update the plots."""
        self.fig.clear()

        plot_type = self.plot_var.get()
        results_list = []
        labels = []

        if self.levy_results:
            results_list.append(self.levy_results)
            labels.append("Levy")
        if self.fit_results:
            results_list.append(self.fit_results)
            labels.append("FIT" if self.fit_valid else "FIT (INVALID)")
        if self.ritz_results:
            results_list.append(self.ritz_results)
            labels.append("Ritz")

        if not results_list:
            self.canvas.draw()
            return

        n_plots = len(results_list)

        if plot_type == '3D':
            for i, (res, label) in enumerate(zip(results_list, labels)):
                ax = self.fig.add_subplot(1, n_plots, i+1, projection='3d')
                ax.plot_surface(res['X'], res['Y'], res['W']*1e3,
                               cmap='viridis', alpha=0.8)
                ax.set_xlabel('x (m)')
                ax.set_ylabel('y (m)')
                ax.set_zlabel('W (mm)')
                ax.set_title(f"{label}: W_max={res['W_max']*1e3:.4f} mm")
        else:
            field_map = {'W': ('W', 1e3, 'mm'), 'Mx': ('Mx', 1, 'N*m/m'), 'vm': ('von_mises', 1e-6, 'MPa')}
            field, scale, unit = field_map.get(plot_type, ('W', 1e3, 'mm'))

            for i, (res, label) in enumerate(zip(results_list, labels)):
                ax = self.fig.add_subplot(1, n_plots, i+1)
                data = res[field] * scale
                cf = ax.contourf(res['X'], res['Y'], data, levels=20, cmap='viridis')
                self.fig.colorbar(cf, ax=ax, label=unit)
                ax.set_xlabel('x (m)')
                ax.set_ylabel('y (m)')
                ax.set_title(f"{label}: max={np.max(np.abs(data)):.4f} {unit}")
                ax.set_aspect('equal')

        self.fig.tight_layout()
        self.canvas.draw()

    def _update_debug(self):
        """Update the debug text."""
        self.debug_text.delete(1.0, tk.END)

        if hasattr(self, 'levy_debug'):
            self.debug_text.insert(tk.END, "=== LEVY DEBUG ===\n")
            for line in self.levy_debug:
                self.debug_text.insert(tk.END, line + "\n")
            self.debug_text.insert(tk.END, "\n")

        if hasattr(self, 'fit_debug'):
            self.debug_text.insert(tk.END, "=== FIT DEBUG ===\n")
            for line in self.fit_debug:
                self.debug_text.insert(tk.END, line + "\n")
            self.debug_text.insert(tk.END, "\n")

        if hasattr(self, 'ritz_debug'):
            self.debug_text.insert(tk.END, "=== RITZ DEBUG ===\n")
            for line in self.ritz_debug:
                self.debug_text.insert(tk.END, line + "\n")

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = TripleMethodGUI()
    app.run()
