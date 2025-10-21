
import os
import io
import json
import uuid
import zipfile
from datetime import datetime

import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, send_file, session, flash

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

import plotly.express as px
import plotly.io as pio

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
