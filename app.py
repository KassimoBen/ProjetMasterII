
import os
import io
import json
import uuid
import zipfile
from datetime import datetime

import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, send_file, session, flash, jsonify

# Optional libs
try:
    from scipy.stats import f_oneway
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

try:
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False

try:
    import cv2
    CV2_OK = True
except Exception:
    CV2_OK = False

import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
from PIL import Image as PILImage, ImageEnhance, ImageFilter

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-key")

# Simple in-memory store per user session
STORE = {}

def get_sid():
    if "sid" not in session:
        import uuid
        session["sid"] = str(uuid.uuid4())
    return session["sid"]

def set_df(df):
    sid = get_sid()
    STORE.setdefault(sid, {})["df"] = df.copy()
    STORE[sid]["original"] = STORE[sid].get("original", df.copy())
    STORE[sid].setdefault("history", [df.copy()])
    STORE[sid].setdefault("future", [])
    STORE[sid].setdefault("pipeline", [])

def get_df():
    sid = get_sid()
    return STORE.get(sid, {}).get("df")

def push_history():
    sid = get_sid()
    STORE[sid]["history"].append(get_df().copy())
    STORE[sid]["future"].clear()

def record_step(action, details=None):
    sid = get_sid()
    STORE[sid]["pipeline"].append({
        "time": datetime.now().isoformat(timespec="seconds"),
        "action": action,
        "details": details or {}
    })

def load_sample(name="Iris"):
    if name == "Iris":
        data = {
            "sepal_length":[5.1,4.9,4.7,6.4,7.3,5.8,6.7,5.6,6.3,5.0],
            "sepal_width":[3.5,3.0,3.2,3.2,2.9,2.7,3.1,2.9,3.3,3.4],
            "petal_length":[1.4,1.4,1.3,4.5,6.3,5.1,5.6,3.6,6.0,1.5],
            "petal_width":[0.2,0.2,0.2,1.5,1.8,1.9,2.4,1.3,2.5,0.2],
            "species":["setosa","setosa","setosa","versicolor","virginica","virginica","virginica","versicolor","virginica","setosa"]
        }
        return pd.DataFrame(data)
    elif name == "Ventes":
        data = {
            "date":["2025-01-01","2025-01-02","2025-01-03","2025-01-04","2025-01-05","2025-01-06"],
            "pays":["MG","MG","MG","FR","FR","FR"],
            "produit":["A","B","A","A","B","B"],
            "quantite":[10,5,7,2,8,9],
            "prix_unitaire":[1000,2000,1000,3000,2500,2500]
        }
        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"])
        df["chiffre_affaires"] = df["quantite"] * df["prix_unitaire"]
        return df
    return None

@app.route("/", methods=["GET", "POST"])
def index():
    user = session.get("user", "")
    if get_df() is None:
        set_df(pd.DataFrame())

    if request.method == "POST":
        action = request.form.get("action")

        if action == "upload":
            file = request.files.get("file")
            sep = request.form.get("sep", ",")
            encoding = request.form.get("encoding", "utf-8")
            sample = request.form.get("sample")
            try:
                if file and file.filename:
                    if file.filename.lower().endswith(".csv"):
                        df = pd.read_csv(file, sep=sep, encoding=encoding)
                    else:
                        df = pd.read_excel(file)
                elif sample and sample != "(aucun)":
                    df = load_sample(sample)
                else:
                    flash("Veuillez sélectionner un fichier ou un jeu d'essai.", "warning")
                    return redirect(url_for("index"))
                set_df(df)
                flash("Données chargées.", "success")
            except Exception as e:
                flash(f"Erreur de lecture: {e}", "danger")
            return redirect(url_for("index"))

        elif action == "undo":
            sid = get_sid()
            hist = STORE[sid]["history"]
            fut = STORE[sid]["future"]
            if len(hist) > 1:
                fut.append(hist.pop())
                STORE[sid]["df"] = hist[-1].copy()

        elif action == "redo":
            sid = get_sid()
            hist = STORE[sid]["history"]
            fut = STORE[sid]["future"]
            if fut:
                nxt = fut.pop()
                hist.append(nxt.copy())
                STORE[sid]["df"] = nxt.copy()

        elif action == "reset":
            sid = get_sid()
            STORE[sid]["df"] = STORE[sid]["original"].copy()
            STORE[sid]["history"] = [STORE[sid]["df"].copy()]
            STORE[sid]["future"].clear()
            STORE[sid]["pipeline"].clear()
            flash("Données réinitialisées.", "info")

        elif action == "dropna_rows_all":
            df = get_df()
            before = len(df)
            df.dropna(how="all", inplace=True)
            set_df(df)
            push_history()
            record_step("dropna_rows_all", {"removed": before - len(df)})
            flash("Lignes vides supprimées.", "success")

        elif action == "dropna_cols_all":
            df = get_df()
            before = df.shape[1]
            df.dropna(axis=1, how="all", inplace=True)
            set_df(df)
            push_history()
            record_step("dropna_cols_all", {"removed": before - df.shape[1]})
            flash("Colonnes vides supprimées.", "success")

        elif action == "drop_duplicates":
            df = get_df()
            before = len(df)
            df.drop_duplicates(inplace=True)
            set_df(df)
            push_history()
            record_step("drop_duplicates", {"removed": before - len(df)})
            flash("Doublons supprimés.", "success")

        elif action == "fillna":
            df = get_df()
            strategy = request.form.get("strategy")
            cols = request.form.getlist("fill_cols")
            if cols and strategy:
                if strategy == "fixed":
                    value = request.form.get("fill_value", "")
                    df[cols] = df[cols].fillna(value)
                    details = {"cols": cols, "strategy": "fixed", "value": value}
                else:
                    for c in cols:
                        if strategy == "mean":
                            val = pd.to_numeric(df[c], errors="coerce").mean()
                        elif strategy == "median":
                            val = pd.to_numeric(df[c], errors="coerce").median()
                        else:
                            m = df[c].mode()
                            val = m.iloc[0] if not m.empty else None
                        df[c] = df[c].fillna(val)
                    details = {"cols": cols, "strategy": strategy}
                set_df(df)
                push_history()
                record_step("fillna", details)
                flash("Valeurs manquantes remplies.", "success")

        elif action == "filter":
            df = get_df()
            expr = request.form.get("query", "")
            try:
                df = df.query(expr)
                set_df(df)
                push_history()
                record_step("filter_query", {"expr": expr})
                flash("Filtre appliqué.", "success")
            except Exception as e:
                flash(f"Requête invalide: {e}", "danger")

        elif action == "select_columns":
            df = get_df()
            cols = request.form.getlist("select_cols")
            if cols:
                df = df[cols]
                set_df(df)
                push_history()
                record_step("select_columns", {"cols": cols})
                flash("Colonnes filtrées.", "success")

        elif action == "rename_column":
            df = get_df()
            old = request.form.get("old_name")
            new = request.form.get("new_name")
            if old and new:
                df.rename(columns={old: new}, inplace=True)
                set_df(df)
                push_history()
                record_step("rename_column", {"old": old, "new": new})
                flash("Colonne renommée.", "success")

        elif action == "astype":
            df = get_df()
            col = request.form.get("astype_col")
            to_type = request.form.get("astype_type")
            try:
                if to_type == "str":
                    df[col] = df[col].astype(str)
                elif to_type == "int":
                    df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
                elif to_type == "float":
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                elif to_type == "datetime":
                    df[col] = pd.to_datetime(df[col], errors="coerce")
                set_df(df)
                push_history()
                record_step("astype", {"col": col, "type": to_type})
                flash("Conversion réussie.", "success")
            except Exception as e:
                flash(f"Erreur de conversion: {e}", "danger")

        elif action == "computed_column":
            df = get_df()
            new_col = request.form.get("new_col")
            expr = request.form.get("expr")
            if new_col and expr:
                try:
                    df[new_col] = pd.eval(expr, engine="python", local_dict={c: df[c] for c in df.columns})
                    set_df(df)
                    push_history()
                    record_step("computed_column", {"new_col": new_col, "expr": expr})
                    flash(f"Colonne '{new_col}' ajoutée.", "success")
                except Exception as e:
                    flash(f"Expression invalide: {e}", "danger")

        elif action == "set_user":
            session["user"] = request.form.get("user_name", "")

        return redirect(url_for("index"))

    df = get_df()
    profile = None
    if df is not None and not df.empty:
        profile = {
            "shape": df.shape,
            "dtypes": df.dtypes.astype(str).to_dict(),
            "missing": df.isna().sum().to_dict()
        }
    sid = get_sid()
    return render_template("index.html",
                           user=user,
                           df=df.head(200) if df is not None else None,
                           cols=df.columns.tolist() if df is not None else [],
                           num_cols=df.select_dtypes(include=[np.number]).columns.tolist() if df is not None else [],
                           cat_cols=df.select_dtypes(exclude=[np.number]).columns.tolist() if df is not None else [],
                           profile=profile,
                           SCIPY_OK=SCIPY_OK,
                           SKLEARN_OK=SKLEARN_OK,
                           pipeline=STORE[sid]["pipeline"])

@app.route("/plot/corr")
def plot_corr():
    df = get_df()
    if df is None or df.empty:
        return "No data", 400
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) < 2:
        return "Need >=2 numeric columns", 400
    corr = df[num_cols].corr()
    fig = px.imshow(corr, text_auto=True, aspect="auto", title="Matrice de corrélation")
    html = pio.to_html(fig, full_html=False, include_plotlyjs="cdn")
    return html


@app.route('/image')
def image_page():
    df = get_df()  # keep compatible usage of STORE
    sid = get_sid()
    imgs = STORE.get(sid, {}).get('images', [])
    active = STORE.get(sid, {}).get('active_image', 0) if imgs else None
    return render_template('image.html', images=imgs, active=active, CV2_OK=CV2_OK)


@app.route('/image/upload', methods=['POST'])
def image_upload():
    files = request.files.getlist('file')
    if not files:
        flash('Aucun fichier sélectionné.', 'warning')
        return redirect(url_for('image_page'))
    allowed = ('.png', '.jpg', '.jpeg')
    max_bytes = 10 * 1024 * 1024
    sid = get_sid()
    STORE.setdefault(sid, {}).setdefault('images', [])
    for file in files:
        if not file or not file.filename:
            continue
        filename = file.filename.lower()
        if not any(filename.endswith(ext) for ext in allowed):
            flash(f'Format non supporté pour {file.filename}. Utilisez PNG ou JPG.', 'danger')
            continue
        data = file.read()
        if len(data) > max_bytes:
            flash(f'Fichier trop volumineux pour {file.filename} (max 10 MB).', 'danger')
            continue
        try:
            img = PILImage.open(io.BytesIO(data)).convert('RGB')
            buf = io.BytesIO()
            img.save(buf, format='PNG')
            preview_bytes = buf.getvalue()
            # store image entry
            imgs = STORE[sid].setdefault('images', [])
            imgs.append({'name': file.filename, 'bytes': data, 'preview': preview_bytes,
                         'original_bytes': data, 'original_preview': preview_bytes,
                         'history': [], 'future': []})
            STORE[sid].setdefault('image_pipeline', [])
        except Exception as e:
            flash(f'Erreur lecture image {file.filename}: {e}', 'danger')
    if STORE[sid].get('images'):
        STORE[sid]['active_image'] = len(STORE[sid]['images']) - 1
        flash('Image(s) chargée(s).', 'success')
    return redirect(url_for('image_page'))


@app.route('/image/undo')
def image_undo():
    sid = get_sid()
    imgs = STORE.get(sid, {}).get('images', [])
    if not imgs:
        flash('Aucune image en mémoire.', 'warning')
        return redirect(url_for('image_page'))
    idx = STORE.get(sid, {}).get('active_image', 0)
    entry = imgs[idx]
    hist = entry.setdefault('history', [])
    fut = entry.setdefault('future', [])
    if not hist:
        flash('Rien à annuler.', 'info')
        return redirect(url_for('image_page'))
    # move current to future and pop last history
    try:
        fut.append(entry.get('bytes'))
        prev = hist.pop()
        entry['bytes'] = prev
        entry['preview'] = prev
        flash('Annulation effectuée.', 'success')
    except Exception as e:
        flash(f'Erreur undo: {e}', 'danger')
    return redirect(url_for('image_page'))


@app.route('/image/redo')
def image_redo():
    sid = get_sid()
    imgs = STORE.get(sid, {}).get('images', [])
    if not imgs:
        flash('Aucune image en mémoire.', 'warning')
        return redirect(url_for('image_page'))
    idx = STORE.get(sid, {}).get('active_image', 0)
    entry = imgs[idx]
    hist = entry.setdefault('history', [])
    fut = entry.setdefault('future', [])
    if not fut:
        flash('Rien à rétablir.', 'info')
        return redirect(url_for('image_page'))
    try:
        nxt = fut.pop()
        hist.append(entry.get('bytes'))
        entry['bytes'] = nxt
        entry['preview'] = nxt
        flash('Rétablissement effectué.', 'success')
    except Exception as e:
        flash(f'Erreur redo: {e}', 'danger')
    return redirect(url_for('image_page'))


@app.route('/image/download_processed')
def image_download_processed():
    sid = get_sid()
    imgs = STORE.get(sid, {}).get('images', [])
    if not imgs:
        return "No image", 404
    idx = STORE.get(sid, {}).get('active_image', 0)
    data = imgs[idx].get('bytes')
    name = imgs[idx].get('name', 'image.png')
    return send_file(io.BytesIO(data), as_attachment=True, download_name=f"processed_{name}", mimetype='image/png')


@app.route('/image/preview_orig/<int:idx>')
def image_preview_orig(idx=0):
    sid = get_sid()
    imgs = STORE.get(sid, {}).get('images', [])
    if not imgs or idx < 0 or idx >= len(imgs):
        return "No image", 404
    img_b = imgs[idx].get('original_preview') or imgs[idx].get('preview')
    return send_file(io.BytesIO(img_b), mimetype='image/png')



@app.route('/image/preview/<int:idx>')
def image_preview(idx=0):
    sid = get_sid()
    imgs = STORE.get(sid, {}).get('images', [])
    if not imgs or idx < 0 or idx >= len(imgs):
        return "No image", 404
    img_b = imgs[idx].get('preview')
    return send_file(io.BytesIO(img_b), mimetype='image/png')


@app.route('/image/select/<int:idx>')
def image_select(idx=0):
    sid = get_sid()
    imgs = STORE.get(sid, {}).get('images', [])
    if not imgs or idx < 0 or idx >= len(imgs):
        flash('Image inexistante.', 'warning')
        return redirect(url_for('image_page'))
    STORE[sid]['active_image'] = idx
    flash(f"Image sélectionnée: {imgs[idx].get('name')}", 'success')
    return redirect(url_for('image_page'))


@app.route('/image/delete/<int:idx>')
def image_delete(idx=0):
    sid = get_sid()
    imgs = STORE.get(sid, {}).get('images', [])
    if not imgs or idx < 0 or idx >= len(imgs):
        flash('Image inexistante.', 'warning')
        return redirect(url_for('image_page'))
    name = imgs[idx].get('name')
    imgs.pop(idx)
    # adjust active
    if imgs:
        STORE[sid]['active_image'] = min(STORE[sid].get('active_image', 0), len(imgs)-1)
    else:
        STORE[sid].pop('active_image', None)
    flash(f'Image supprimée: {name}', 'info')
    return redirect(url_for('image_page'))


@app.route('/image/hist')
def image_hist():
    sid = get_sid()
    imgs = STORE.get(sid, {}).get('images', [])
    if not imgs:
        return "<p>Aucune image chargée.</p>", 400
    active = STORE.get(sid, {}).get('active_image', 0)
    img_b = imgs[active]['bytes']
    try:
        img = PILImage.open(io.BytesIO(img_b)).convert('RGB')
        arr = np.array(img)
        # compute hist per channel
        bins = list(range(257))
        r, _ = np.histogram(arr[:, :, 0].flatten(), bins=bins)
        g, _ = np.histogram(arr[:, :, 1].flatten(), bins=bins)
        b, _ = np.histogram(arr[:, :, 2].flatten(), bins=bins)
        x = list(range(256))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=r, mode='lines', name='R', line=dict(color='red')))
        fig.add_trace(go.Scatter(x=x, y=g, mode='lines', name='G', line=dict(color='green')))
        fig.add_trace(go.Scatter(x=x, y=b, mode='lines', name='B', line=dict(color='blue')))
        fig.update_layout(title='Histogramme couleurs', xaxis_title='Intensité', yaxis_title='Comptage')
        html = pio.to_html(fig, full_html=False, include_plotlyjs='cdn')
        return html
    except Exception as e:
        return f"<p>Erreur calcul histogramme: {e}</p>", 500


@app.route('/image/measure')
def image_measure():
    """Compute contours and measurements for the active image and return JSON (or CSV if format=csv)."""
    if not CV2_OK:
        msg = ('OpenCV non installé, mesures indisponibles.\n'
               'Pour activer les mesures installez OpenCV dans votre environnement:\n'
               'pip install opencv-python')
        return jsonify(error=msg), 400
    sid = get_sid()
    imgs = STORE.get(sid, {}).get('images', [])
    if not imgs:
        return jsonify(error='Aucune image chargée.'), 400
    idx = STORE.get(sid, {}).get('active_image', 0)
    entry = imgs[idx]
    img_b = entry.get('bytes')
    try:
        # decode image
        nparr = np.frombuffer(img_b, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # parameters: method and retrieval mode
        method = request.args.get('method', 'canny')
        retrieval = request.args.get('retr', 'external')
        retr_mode = cv2.RETR_EXTERNAL if retrieval == 'external' else cv2.RETR_TREE

        # find contours based on chosen method
        if method == 'canny':
            low = int(request.args.get('low', 50))
            high = int(request.args.get('high', 150))
            edges = cv2.Canny(gray, low, high)
            cnts, _ = cv2.findContours(edges, retr_mode, cv2.CHAIN_APPROX_SIMPLE)
        else:
            thresh_type = request.args.get('thresh_type', 'otsu')
            if thresh_type == 'otsu':
                _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            else:
                tval = int(request.args.get('thresh', 128))
                _, th = cv2.threshold(gray, tval, 255, cv2.THRESH_BINARY)
            cnts, _ = cv2.findContours(th, retr_mode, cv2.CHAIN_APPROX_SIMPLE)

        # filtering parameters
        try:
            min_area = float(request.args.get('min_area', 0.0))
        except Exception:
            min_area = 0.0
        try:
            min_perim = float(request.args.get('min_perimeter', 0.0))
        except Exception:
            min_perim = 0.0
        try:
            approx_eps = float(request.args.get('approx_eps', 0.0))
        except Exception:
            approx_eps = 0.0
        try:
            min_points = int(request.args.get('min_points', 0))
        except Exception:
            min_points = 0
        try:
            max_contours = int(request.args.get('max_contours', 0))
        except Exception:
            max_contours = 0
        sort_by = request.args.get('sort_by', 'area')

        measures = []
        filtered_cnts = []
        # iterate and filter
        for c in cnts:
            area = float(cv2.contourArea(c))
            peri = float(cv2.arcLength(c, True))
            # approximation if requested (eps is absolute pixels)
            c_approx = c
            pts = int(c.shape[0])
            if approx_eps and approx_eps > 0:
                try:
                    c_approx = cv2.approxPolyDP(c, approx_eps, True)
                    pts = int(c_approx.shape[0])
                except Exception:
                    c_approx = c
                    pts = int(c.shape[0])

            # apply filters
            if area < min_area or peri < min_perim or (min_points and pts < min_points):
                continue

            x, y, w, h = cv2.boundingRect(c_approx)
            measures.append({'area': area, 'perimeter': peri, 'bbox': [int(x), int(y), int(w), int(h)], 'points': pts, 'raw_contour': c_approx})
            filtered_cnts.append(c_approx)

        total_found = len(cnts)
        # sort and limit
        order = list(range(len(measures)))
        if sort_by == 'perimeter':
            order.sort(key=lambda i: measures[i]['perimeter'], reverse=True)
        else:
            order.sort(key=lambda i: measures[i]['area'], reverse=True)
        if max_contours and max_contours > 0:
            order = order[:max_contours]

        # build overlay: draw all original contours lightly, then selected contours prominently
        overlay = img.copy()
        try:
            # draw all contours faintly
            cv2.drawContours(overlay, cnts, -1, (150, 150, 150), 1)
        except Exception:
            pass
        selected = []
        for out_idx, mi in enumerate(order):
            c = measures[mi]['raw_contour']
            cv2.drawContours(overlay, [c], -1, (0, 255, 0), 2)
            # put a small label at centroid
            M = cv2.moments(c)
            if M.get('m00', 0):
                cx = int(M['m10'] / M['m00']); cy = int(M['m01'] / M['m00'])
                cv2.putText(overlay, str(out_idx), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            selected.append({'id': out_idx, 'area': measures[mi]['area'], 'perimeter': measures[mi]['perimeter'], 'bbox': measures[mi]['bbox'], 'points': measures[mi]['points']})

        # encode overlay
        _, buf = cv2.imencode('.png', overlay)
        overlay_bytes = buf.tobytes()
        entry['last_overlay'] = overlay_bytes

        fmt = request.args.get('format', '').lower()
        if fmt == 'csv':
            import csv
            out = io.StringIO()
            writer = csv.writer(out)
            writer.writerow(['id', 'area', 'perimeter', 'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h', 'points'])
            for m in selected:
                writer.writerow([m['id'], m['area'], m['perimeter'], *m['bbox'], m.get('points', '')])
            return send_file(io.BytesIO(out.getvalue().encode('utf-8')), as_attachment=True, download_name='measures.csv', mimetype='text/csv')

        return jsonify(measures=selected, overlay_url=url_for('image_measure_overlay'), total_found=total_found, returned=len(selected))
    except Exception as e:
        return jsonify(error=str(e)), 500


@app.route('/image/measure_overlay')
def image_measure_overlay():
    sid = get_sid()
    imgs = STORE.get(sid, {}).get('images', [])
    if not imgs:
        return "No image", 404
    idx = STORE.get(sid, {}).get('active_image', 0)
    entry = imgs[idx]
    data = entry.get('last_overlay')
    if not data:
        return "No overlay", 404
    return send_file(io.BytesIO(data), mimetype='image/png')


@app.route('/image/preprocess', methods=['POST'])
def image_preprocess():
    """Apply preprocessing operation(s) to the currently active image.
    Accepts form-data with fields:
      - op: operation name (see supported_ops)
      - params: JSON string or individual form fields for parameters
    Returns: redirect back to image page or JSON when requested.
    """
    sid = get_sid()
    imgs = STORE.get(sid, {}).get('images', [])
    if not imgs:
        flash('Aucune image chargée.', 'warning')
        return redirect(url_for('image_page'))

    idx = STORE.get(sid, {}).get('active_image', 0)
    if idx is None or idx < 0 or idx >= len(imgs):
        flash('Image active invalide.', 'warning')
        return redirect(url_for('image_page'))

    op = request.form.get('op') or (request.get_json(silent=True) or {}).get('op')
    # accept params as JSON string or form fields
    params = {}
    params_raw = request.form.get('params')
    if params_raw:
        try:
            params = json.loads(params_raw)
        except Exception:
            params = {}
    # also merge individual params
    for k, v in request.form.items():
        if k not in ('op', 'params'):
            params[k] = v

    # load image into OpenCV (BGR)
    img_b = imgs[idx]['bytes']
    try:
        if CV2_OK:
            nparr = np.frombuffer(img_b, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        else:
            pil = PILImage.open(io.BytesIO(img_b)).convert('RGB')
            img = np.array(pil)
            if CV2_OK:
                img = img[:, :, ::-1]
    except Exception as e:
        flash(f'Erreur lecture image: {e}', 'danger')
        return redirect(url_for('image_page'))

    # helper converters
    def to_gray(im):
        if CV2_OK:
            return cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        else:
            return np.array(PILImage.fromarray(im).convert('L'))

    def save_back(im_arr):
        # im_arr expected as BGR or grayscale
        try:
            if isinstance(im_arr, np.ndarray):
                if im_arr.ndim == 2:
                    out = PILImage.fromarray(im_arr)
                else:
                    rgb = im_arr[:, :, ::-1] if CV2_OK else im_arr
                    out = PILImage.fromarray(rgb)
                buf = io.BytesIO()
                out.save(buf, format='PNG')
                data = buf.getvalue()
                # push to history before overwriting
                imgs[idx].setdefault('history', [])
                imgs[idx].setdefault('future', [])
                try:
                    prev = imgs[idx].get('bytes')
                    if prev is not None:
                        imgs[idx]['history'].append(prev)
                except Exception:
                    pass
                imgs[idx]['bytes'] = data
                # update preview
                imgs[idx]['preview'] = data
                # clear redo stack
                imgs[idx]['future'].clear()
                # record pipeline
                STORE[sid].setdefault('image_pipeline', [])
                STORE[sid]['image_pipeline'].append({'time': datetime.now().isoformat(timespec='seconds'), 'op': op, 'params': params})
                flash(f"Opération '{op}' appliquée.", 'success')
                return True
        except Exception as e:
            flash(f'Erreur sauvegarde image: {e}', 'danger')
        return False

    # supported ops
    try:
        if not op:
            flash('Opération non spécifiée.', 'warning')
            return redirect(url_for('image_page'))

        # numeric helper
        def _int(k, default=0):
            try:
                return int(params.get(k, default))
            except Exception:
                return default

        if op == 'grayscale':
            gray = to_gray(img)
            save_back(gray)

        elif op == 'resize':
            w = _int('width', 0)
            h = _int('height', 0)
            if w <= 0 or h <= 0:
                flash('width et height requis pour resize.', 'warning')
            else:
                if CV2_OK:
                    out = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
                else:
                    pil = PILImage.open(io.BytesIO(img_b)).convert('RGB')
                    out = np.array(pil.resize((w, h)))
                save_back(out)

        elif op == 'rotate':
            angle = float(params.get('angle', 0))
            pil = PILImage.open(io.BytesIO(img_b)).convert('RGB')
            out = np.array(pil.rotate(angle, expand=True))
            save_back(out)

        elif op == 'flip':
            mode = params.get('mode', 'h')
            if CV2_OK:
                if mode == 'h':
                    out = cv2.flip(img, 1)
                else:
                    out = cv2.flip(img, 0)
            else:
                pil = PILImage.open(io.BytesIO(img_b)).convert('RGB')
                if mode == 'h':
                    out = np.array(pil.transpose(PILImage.FLIP_LEFT_RIGHT))
                else:
                    out = np.array(pil.transpose(PILImage.FLIP_TOP_BOTTOM))
            save_back(out)

        elif op == 'blur_gaussian':
            k = _int('ksize', 5)
            if k % 2 == 0: k += 1
            if CV2_OK:
                out = cv2.GaussianBlur(img, (k, k), 0)
            else:
                pil = PILImage.open(io.BytesIO(img_b)).convert('RGB')
                out = np.array(pil.filter(ImageFilter.GaussianBlur(radius=k)))
            save_back(out)

        elif op == 'blur_median':
            k = _int('ksize', 5)
            if k % 2 == 0: k += 1
            if CV2_OK:
                out = cv2.medianBlur(img, k)
                save_back(out)
            else:
                flash('Median blur requires OpenCV.', 'warning')

        elif op == 'bilateral':
            d = _int('d', 9)
            sigmaColor = _int('sigmaColor', 75)
            sigmaSpace = _int('sigmaSpace', 75)
            if CV2_OK:
                out = cv2.bilateralFilter(img, d, sigmaColor, sigmaSpace)
                save_back(out)
            else:
                flash('Bilateral filter requires OpenCV.', 'warning')

        elif op == 'sharpen':
            if CV2_OK:
                kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
                out = cv2.filter2D(img, -1, kernel)
                save_back(out)
            else:
                flash('Sharpen requires OpenCV.', 'warning')

        elif op == 'clahe':
            clip = float(params.get('clipLimit', 2.0))
            tiles = int(params.get('tileGrid', 8))
            if CV2_OK:
                lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tiles, tiles))
                cl = clahe.apply(l)
                merged = cv2.merge((cl, a, b))
                out = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
                save_back(out)
            else:
                flash('CLAHE requires OpenCV.', 'warning')

        elif op == 'contrast_brightness':
            alpha = float(params.get('alpha', 1.0))
            beta = float(params.get('beta', 0.0))
            if CV2_OK:
                out = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
                save_back(out)
            else:
                pil = PILImage.open(io.BytesIO(img_b)).convert('RGB')
                enhancer = ImageEnhance.Contrast(pil)
                out_p = enhancer.enhance(alpha)
                out = np.array(out_p)
                save_back(out)

        elif op == 'gamma':
            g = float(params.get('gamma', 1.0))
            invGamma = 1.0 / g if g != 0 else 1.0
            table = np.array([((i / 255.0) ** invGamma) * 255 for i in range(256)]).astype('uint8')
            if CV2_OK:
                out = cv2.LUT(img, table)
                save_back(out)
            else:
                arr = np.array(PILImage.open(io.BytesIO(img_b)).convert('RGB'))
                out = table[arr]
                save_back(out)

        elif op == 'denoise':
            if CV2_OK:
                out = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
                save_back(out)
            else:
                flash('Denoise requires OpenCV.', 'warning')

        elif op == 'canny':
            low = _int('low', 50)
            high = _int('high', 150)
            gray = to_gray(img)
            if CV2_OK:
                edges = cv2.Canny(gray, low, high)
                save_back(edges)
            else:
                flash('Canny requires OpenCV.', 'warning')

        elif op == 'otsu':
            gray = to_gray(img)
            if CV2_OK:
                _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                save_back(th)
            else:
                flash('Otsu threshold requires OpenCV.', 'warning')

        else:
            flash('Opération non supportée.', 'warning')
    except Exception as e:
        flash(f'Erreur lors du traitement: {e}', 'danger')

    return redirect(url_for('image_page'))
    return redirect(url_for('image_page'))


def _process_image_bytes(img_b, op, params):
    """Apply a preview-only processing to image bytes and return PNG bytes.
    Supports a safe subset of operations used for live preview.
    """
    try:
        pil = PILImage.open(io.BytesIO(img_b)).convert('RGB')
    except Exception as e:
        raise

    def to_bytes(pil_img):
        buf = io.BytesIO()
        pil_img.save(buf, format='PNG')
        return buf.getvalue()

    # convenience getters
    def get_int(k, default=None):
        v = params.get(k)
        if v is None or v == '':
            return default
        try:
            return int(v)
        except Exception:
            try:
                return int(float(v))
            except Exception:
                return default

    def get_float(k, default=None):
        v = params.get(k)
        if v is None or v == '':
            return default
        try:
            return float(v)
        except Exception:
            return default

    op = (op or '').lower()
    # grayscale
    if op == 'grayscale':
        out = pil.convert('L').convert('RGB')
        return to_bytes(out)

    if op == 'resize':
        w = get_int('width')
        h = get_int('height')
        if w and h:
            out = pil.resize((w, h), PILImage.LANCZOS)
            return to_bytes(out)

    if op == 'rotate':
        angle = get_float('angle', 0) or 0
        out = pil.rotate(angle, expand=True)
        return to_bytes(out)

    if op == 'flip':
        mode = (params.get('mode') or 'h').lower()
        if mode == 'v':
            out = pil.transpose(PILImage.FLIP_TOP_BOTTOM)
        else:
            out = pil.transpose(PILImage.FLIP_LEFT_RIGHT)
        return to_bytes(out)

    if op == 'blur_gaussian':
        k = get_float('ksize', 2)
        out = pil.filter(ImageFilter.GaussianBlur(radius=k))
        return to_bytes(out)

    if op == 'contrast_brightness':
        alpha = get_float('alpha', 1.0) or 1.0
        beta = get_float('beta', 0.0) or 0.0
        enh = ImageEnhance.Contrast(pil)
        out = enh.enhance(alpha)
        if beta != 0:
            be = ImageEnhance.Brightness(out)
            out = be.enhance(1.0 + (beta / 100.0))
        return to_bytes(out)

    if op == 'gamma':
        g = get_float('gamma', 1.0) or 1.0
        inv = 1.0 / g if g != 0 else 1.0
        lut = [int(max(0, min(255, pow(i / 255.0, inv) * 255))) for i in range(256)]
        out = pil.point(lut * 3)
        return to_bytes(out)

    # OpenCV-backed ops
    if 'cv2' in globals() and CV2_OK:
        arr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        if op == 'blur_median':
            k = get_int('ksize', 3) or 3
            if k % 2 == 0: k += 1
            out_arr = cv2.medianBlur(arr, k)
            out = PILImage.fromarray(cv2.cvtColor(out_arr, cv2.COLOR_BGR2RGB))
            return to_bytes(out)

        if op == 'bilateral':
            d = get_int('d', 9) or 9
            sigmaColor = get_int('sigmaColor', 75) or 75
            sigmaSpace = get_int('sigmaSpace', 75) or 75
            out_arr = cv2.bilateralFilter(arr, d, sigmaColor, sigmaSpace)
            out = PILImage.fromarray(cv2.cvtColor(out_arr, cv2.COLOR_BGR2RGB))
            return to_bytes(out)

        if op == 'clahe':
            clip = get_float('clipLimit', 2.0) or 2.0
            tiles = get_int('tileGrid', 8) or 8
            lab = cv2.cvtColor(arr, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tiles, tiles))
            cl = clahe.apply(l)
            merged = cv2.merge((cl, a, b))
            out_arr = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
            out = PILImage.fromarray(cv2.cvtColor(out_arr, cv2.COLOR_BGR2RGB))
            return to_bytes(out)

        if op == 'denoise':
            h = get_float('h', 10) or 10
            out_arr = cv2.fastNlMeansDenoisingColored(arr, None, h, h, 7, 21)
            out = PILImage.fromarray(cv2.cvtColor(out_arr, cv2.COLOR_BGR2RGB))
            return to_bytes(out)

        if op == 'canny':
            low = get_int('low', 50) or 50
            high = get_int('high', 150) or 150
            gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, low, high)
            out = PILImage.fromarray(edges).convert('RGB')
            return to_bytes(out)

        if op == 'otsu':
            gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
            _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            out = PILImage.fromarray(th).convert('RGB')
            return to_bytes(out)

    # default: return original
    return to_bytes(pil)


@app.route('/image/preview_process', methods=['POST'])
def image_preview_process():
    """API for live preview: accepts JSON {op: ..., params: {...}} and returns image/png bytes without saving."""
    sid = get_sid()
    imgs = STORE.get(sid, {}).get('images', [])
    if not imgs:
        return jsonify(error='Aucune image chargée.'), 400
    active = STORE.get(sid, {}).get('active_image', 0)
    if active is None or active < 0 or active >= len(imgs):
        return jsonify(error='Image active invalide.'), 400
    img_b = imgs[active]['bytes']
    try:
        payload = request.get_json(force=True)
    except Exception:
        payload = {}
    op = payload.get('op')
    params = payload.get('params', {}) or {}
    try:
        out_bytes = _process_image_bytes(img_b, op, params)
        return send_file(io.BytesIO(out_bytes), mimetype='image/png')
    except Exception as e:
        return jsonify(error=str(e)), 500

@app.route("/plot/general")
def plot_general():
    df = get_df()
    df = get_df()
    if df is None or df.empty:
        return "<p>Donn\u00e9es insuffisantes.</p>", 400

    plot_type = request.args.get("type", "scatter")
    x_col = request.args.get("x")
    y_col = request.args.get("y")
    z_col = request.args.get("z")
    color_col = request.args.get("color")

    # histogram only requires x_col
    if plot_type == "histogram":
        if not x_col:
            return "<p>Colonne X requise pour histogramme.</p>", 400
    else:
        # for 3D plots require z
        if plot_type in ("scatter3d","line3d"):
            if not x_col or not y_col or not z_col:
                return "<p>Colonnes X, Y et Z requises pour 3D.</p>", 400
        else:
            if not x_col or not y_col:
                return "<p>Colonnes X et Y requises.</p>", 400

    try:
        df_plot = df.copy()
        mappings = {}

        # Helper: treat histogram of categorical values as bar of counts
        if plot_type == "histogram":
            col = df_plot[x_col]
            if pd.api.types.is_numeric_dtype(col) or pd.api.types.is_datetime64_any_dtype(col):
                fig = px.histogram(df_plot, x=x_col, color=color_col, title=f"Histogramme: {x_col}")
            else:
                counts = col.astype(str).value_counts().reset_index()
                counts.columns = [x_col, 'count']
                fig = px.bar(counts, x=x_col, y='count', title=f"Histogramme (cat\u00e9g.): {x_col}")
        else:
            # Prepare X
            x_ser = df_plot[x_col]
            # If datetime-like, convert
            if pd.api.types.is_datetime64_any_dtype(x_ser):
                df_plot['_x'] = x_ser
                x_use = '_x'
            else:
                # try to parse datetimes
                try:
                    x_dt = pd.to_datetime(x_ser, errors='coerce')
                except Exception:
                    x_dt = pd.Series([pd.NaT]*len(x_ser))
                if x_dt.notna().sum() > 0 and x_dt.isna().sum() < len(x_dt):
                    df_plot['_x'] = x_dt
                    x_use = '_x'
                else:
                    # keep as-is but factorize strings to numeric codes for axes if needed
                    if pd.api.types.is_numeric_dtype(x_ser):
                        df_plot['_x'] = x_ser
                        x_use = '_x'
                    else:
                        codes, uniques = pd.factorize(x_ser.astype(str))
                        df_plot['_x'] = codes
                        mappings['x'] = list(uniques)
                        x_use = '_x'

            # Prepare Y
            y_ser = df_plot[y_col]
            if pd.api.types.is_numeric_dtype(y_ser):
                df_plot['_y'] = pd.to_numeric(y_ser, errors='coerce')
            else:
                # factorize non-numeric Y so we can still plot; keep mapping
                codes, uniques = pd.factorize(y_ser.astype(str))
                df_plot['_y'] = codes
                mappings['y'] = list(uniques)

            # Build figure with transformed columns
            if plot_type == 'scatter':
                fig = px.scatter(df_plot, x=x_use, y='_y', color=color_col if color_col else None,
                                 title=f"Nuage de points: {x_col} vs {y_col}")
            elif plot_type == 'line':
                fig = px.line(df_plot, x=x_use, y='_y', color=color_col if color_col else None,
                              title=f"Ligne: {x_col} vs {y_col}")
            elif plot_type == 'scatter3d' or plot_type == 'line3d':
                # Prepare Z
                z_ser = df_plot[z_col]
                if pd.api.types.is_numeric_dtype(z_ser):
                    df_plot['_z'] = pd.to_numeric(z_ser, errors='coerce')
                else:
                    codes, uniques = pd.factorize(z_ser.astype(str))
                    df_plot['_z'] = codes
                    mappings['z'] = list(uniques)
                if plot_type == 'scatter3d':
                    fig = px.scatter_3d(df_plot, x=x_use, y='_y', z='_z', color=color_col if color_col else None,
                                        title=f"Nuage 3D: {x_col} vs {y_col} vs {z_col}")
                else:
                    fig = px.line_3d(df_plot, x=x_use, y='_y', z='_z', color=color_col if color_col else None,
                                     title=f"Ligne 3D: {x_col} vs {y_col} vs {z_col}")
            elif plot_type == 'bar':
                fig = px.bar(df_plot, x=x_use, y='_y', color=color_col if color_col else None,
                             title=f"Barres: {x_col} vs {y_col}")
            elif plot_type == 'box':
                # box accepts categorical x; use transformed x and y
                fig = px.box(df_plot, x=x_use, y='_y', color=color_col if color_col else None,
                             title=f"Bo\u00eete \u00e0 moustaches: {x_col} vs {y_col}")
            else:
                return "<p>Type de graphique non support\u00e9.</p>", 400

            # Apply tick label mappings for factorized columns so users see original categories
            if 'x' in mappings:
                uniques = mappings['x']
                fig.update_xaxes(tickmode='array', tickvals=list(range(len(uniques))), ticktext=uniques)
            if 'y' in mappings:
                uniques = mappings['y']
                fig.update_yaxes(tickmode='array', tickvals=list(range(len(uniques))), ticktext=uniques)

        # Serialize with Plotly's to_json (handles numpy types)
        fig_json = fig.to_json()
        fig_obj = json.loads(fig_json)
        return jsonify(fig=fig_obj)
    except Exception as e:
        return f"<p>Erreur de g\u00e9n\u00e9ration: {e}</p>", 400
@app.route("/analyse/anova")
def anova():
    if not SCIPY_OK:
        return "<p>scipy non installé.</p>"
    df = get_df()
    if df is None or df.empty:
        return "<p>Données insuffisantes.</p>"
    y = request.args.get("y")
    g = request.args.get("g")
    if not y or not g:
        return "<p>Paramètres manquants.</p>"
    groups = [grp[y].dropna().values for _, grp in df.groupby(g)]
    if len(groups) < 2:
        return "<p>Au moins 2 groupes requis.</p>"
    F, p = f_oneway(*groups)
    return f"<p>F = {F:.4f}, p = {p:.4e}</p>"

@app.route("/analyse/regression")
def regression():
    if not SKLEARN_OK:
        return "<p>scikit-learn non installé.</p>"
    df = get_df()
    if df is None or df.empty:
        return "<p>Données insuffisantes.</p>"
    target = request.args.get("y")
    feats = request.args.get("x", "")
    features = [f for f in feats.split(",") if f]
    if not target or not features:
        return "<p>Paramètres manquants.</p>"
    X = df[features].fillna(0).values
    y = df[target].fillna(0).values
    model = LinearRegression().fit(X, y)
    r2 = model.score(X, y)
    coefs = {features[i]: float(model.coef_[i]) for i in range(len(features))}
    return f"<p>R² = {r2:.4f}</p><pre>{json.dumps({'coefficients': coefs, 'intercept': float(model.intercept_)}, indent=2)}</pre>"

@app.route("/stats/column")
def stats_column():
    """Return descriptive statistics and a small Plotly figure (as JSON) for a given column.
    Args (query): col, plot (hist|box|none)
    """
    df = get_df()
    if df is None or df.empty:
        return jsonify(error="Donn\u00e9es insuffisantes."), 400
    col = request.args.get('col')
    plot = request.args.get('plot', 'hist')
    if not col or col not in df.columns:
        return jsonify(error='Colonne manquante ou inexistante.'), 400

    ser = df[col]
    stats = {
        'count': int(ser.count()),
        'missing': int(ser.isna().sum()),
    }

    # add numeric summaries when possible
    if pd.api.types.is_numeric_dtype(ser):
        snum = pd.to_numeric(ser, errors='coerce')
        stats.update({
            'mean': float(snum.mean()) if not snum.dropna().empty else None,
            'median': float(snum.median()) if not snum.dropna().empty else None,
            'std': float(snum.std()) if not snum.dropna().empty else None,
            'min': float(snum.min()) if not snum.dropna().empty else None,
            'max': float(snum.max()) if not snum.dropna().empty else None,
        })
    else:
        # attempt datetime bounds
        try:
            sdt = pd.to_datetime(ser, errors='coerce')
            if sdt.notna().sum() > 0:
                stats.update({'min': str(sdt.min()), 'max': str(sdt.max())})
            else:
                stats.update({'unique': int(ser.astype(str).nunique())})
        except Exception:
            stats.update({'unique': int(ser.astype(str).nunique())})

    fig_obj = None
    try:
        if plot == 'hist':
            if pd.api.types.is_numeric_dtype(ser) or pd.api.types.is_datetime64_any_dtype(ser):
                fig = px.histogram(df, x=col, title=f"Histogramme: {col}")
            else:
                counts = ser.astype(str).value_counts().reset_index()
                counts.columns = [col, 'count']
                fig = px.bar(counts, x=col, y='count', title=f"Distribution: {col}")
        elif plot == 'box':
            if pd.api.types.is_numeric_dtype(ser):
                fig = px.box(df, y=col, title=f"Bo\u00eete: {col}")
            else:
                counts = ser.astype(str).value_counts().reset_index()
                counts.columns = [col, 'count']
                fig = px.bar(counts, x=col, y='count', title=f"Distribution: {col}")
        else:
            fig = None

        if fig is not None:
            try:
                fig_obj = json.loads(fig.to_json())
            except Exception:
                fig_obj = fig.to_dict()
    except Exception:
        fig_obj = None

    return jsonify(stats=stats, fig=fig_obj)


@app.route("/predict", methods=["POST"])
def predict():
    """Train a simple LinearRegression on selected features and return coefficients and optional prediction.
    Expects JSON body: { 'y': 'target_col', 'x': ['f1','f2'], 'values': { 'f1': val1, 'f2': val2 } }
    """
    if not SKLEARN_OK:
        return jsonify(error='scikit-learn non installé.'), 400
    df = get_df()
    if df is None or df.empty:
        return jsonify(error='Données insuffisantes.'), 400
    try:
        payload = request.get_json(force=True)
    except Exception:
        return jsonify(error='JSON invalide.'), 400
    y_col = payload.get('y')
    x_cols = payload.get('x', [])
    values = payload.get('values')
    if not y_col or not x_cols:
        return jsonify(error='y et x requis.'), 400

    # Prepare training data: handle categoricals by factorize
    X_df = df[x_cols].copy()
    mapping = {}
    for c in X_df.columns:
        if not pd.api.types.is_numeric_dtype(X_df[c]):
            codes, uniques = pd.factorize(X_df[c].astype(str))
            X_df[c] = codes
            mapping[c] = list(uniques)
    # Target
    y_ser = df[y_col]
    if not pd.api.types.is_numeric_dtype(y_ser):
        # try to coerce
        try:
            y_train = pd.to_numeric(y_ser, errors='coerce').fillna(0).values
        except Exception:
            y_train = y_ser.astype('category').cat.codes.values
    else:
        y_train = pd.to_numeric(y_ser, errors='coerce').fillna(0).values

    X_train = X_df.fillna(0).values
    model = LinearRegression().fit(X_train, y_train)
    coefs = {x_cols[i]: float(model.coef_[i]) for i in range(len(x_cols))}
    intercept = float(model.intercept_)
    res = {'r2': model.score(X_train, y_train), 'coefficients': coefs, 'intercept': intercept}

    # If values provided, prepare a single-row input for prediction
    if values:
        row = []
        for c in x_cols:
            v = values.get(c)
            if c in mapping:
                # map categorical value to code if exists, else -1
                try:
                    code = mapping[c].index(str(v))
                except ValueError:
                    code = -1
                row.append(code)
            else:
                try:
                    row.append(float(v))
                except Exception:
                    row.append(0.0)
        try:
            pred = model.predict([row])[0]
            res['prediction'] = float(pred)
        except Exception as e:
            res['prediction_error'] = str(e)

    return jsonify(res)

@app.route("/pivot")
def pivot():
    df = get_df()
    if df is None or df.empty:
        return "<p>Données insuffisantes.</p>"
    values = request.args.get("values", "")
    index = request.args.get("index", "")
    columns = request.args.get("columns", "")
    agg = request.args.get("agg", "sum")
    vals = [v for v in values.split(",") if v]
    idx = [v for v in index.split(",") if v]
    cols = [v for v in columns.split(",") if v]
    if not vals or not idx:
        return "<p>Choisissez au moins Valeurs et Index.</p>"
    pivot = pd.pivot_table(df, values=vals, index=idx, columns=cols if cols else None, aggfunc=agg, fill_value=0)
    return pivot.to_html(classes="table table-striped table-sm")

@app.route("/normalize")
def normalize():
    if not SKLEARN_OK:
        return "<p>scikit-learn non installé.</p>"
    df = get_df()
    if df is None or df.empty:
        return "<p>Données insuffisantes.</p>"
    method = request.args.get("method", "minmax")
    cols = [c for c in request.args.get("cols","").split(",") if c]
    if not cols:
        return "<p>Aucune colonne.</p>"
    if method == "minmax":
        scaler = MinMaxScaler()
    elif method == "standard":
        scaler = StandardScaler()
    else:
        scaler = RobustScaler()
    df[cols] = scaler.fit_transform(df[cols])
    set_df(df)
    push_history()
    record_step("normalize", {"cols": cols, "method": method})
    return "<p>OK</p>"

@app.route("/calculate")
def calculate():
    df = get_df()
    if df is None or df.empty:
        return "<p>Données insuffisantes.</p>", 400
    expr = request.args.get("expr", "")
    if not expr:
        return "<p>Expression requise.</p>", 400
    try:
        result = pd.eval(expr, engine="python", local_dict={c: df[c] for c in df.columns})
        if isinstance(result, (int, float)):
            return f"<p>Résultat: {result}</p>"
        elif hasattr(result, 'shape'):
            return f"<p>Résultat: {result.shape[0]} éléments</p><pre>{result.to_string()}</pre>"
        else:
            return f"<p>Résultat: {result}</p>"
    except Exception as e:
        return f"<p>Erreur: {e}</p>", 400

@app.route("/download/<fmt>")
def download(fmt):
    df = get_df()
    if df is None or df.empty:
        return "No data", 400
    if fmt == "csv":
        data = df.to_csv(index=False).encode("utf-8")
        return send_file(io.BytesIO(data), as_attachment=True, download_name="donnees_nettoyees.csv", mimetype="text/csv")
    elif fmt == "excel":
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="data")
        buffer.seek(0)
        return send_file(buffer, as_attachment=True, download_name="donnees_nettoyees.xlsx", mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    elif fmt == "json":
        data = df.to_json(orient="records", force_ascii=False).encode("utf-8")
        return send_file(io.BytesIO(data), as_attachment=True, download_name="donnees_nettoyees.json", mimetype="application/json")
    return "Bad format", 400

@app.route("/download_project")
def download_project():
    df = get_df()
    if df is None:
        return "No data", 400
    sid = get_sid()
    pipeline = STORE[sid]["pipeline"]
    meta = {"created": datetime.now().isoformat(timespec="seconds"), "user": session.get("user","")}
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("data.csv", df.to_csv(index=False))
        z.writestr("pipeline.json", json.dumps(pipeline, ensure_ascii=False, indent=2))
        z.writestr("meta.json", json.dumps(meta, ensure_ascii=False, indent=2))
    buffer.seek(0)
    return send_file(buffer, as_attachment=True, download_name="projet_datalab.zip", mimetype="application/zip")

@app.template_filter("head")
def head_filter(df, n=100):
    try:
        return df.head(n).to_html(classes="table table-hover table-sm", index=False)
    except Exception:
        return "<p>(Aperçu indisponible)</p>"

if __name__ == "__main__":
    app.run(debug=True)
