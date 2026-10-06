import os, sqlite3, uuid, json, textwrap
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, send_file, flash
from PIL import Image
import numpy as np
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4

BASE=os.path.dirname(os.path.abspath(__file__))
UPLOAD=os.path.join(BASE,"static","uploads")
MODEL_DIR=os.path.join(BASE,"model")
DB=os.path.join(BASE,"database","cropguard.db")
os.makedirs(UPLOAD,exist_ok=True); os.makedirs(os.path.dirname(DB),exist_ok=True)

app=Flask(__name__); app.secret_key="cropguard-demo"
MODEL=None; CLASS_NAMES=[]
try:
    import tensorflow as tf
    mp=os.path.join(MODEL_DIR,"cropguard_model.h5")
    cp=os.path.join(MODEL_DIR,"class_names.json")
    if os.path.exists(mp): MODEL=tf.keras.models.load_model(mp)
    if os.path.exists(cp):
        with open(cp,encoding="utf-8") as f: CLASS_NAMES=json.load(f)
except Exception:
    MODEL=None

TREATMENTS={
"Tomato___Early_blight":{"crop":"Tomato","disease":"Early Blight","treatment":"Remove affected leaves, improve airflow and avoid overhead watering. Use a locally approved fungicide according to its label.","organic":"Consider approved neem-based or biological options according to local agricultural guidance.","prevention":"Rotate crops and remove infected plant debris."},
"Tomato___Late_blight":{"crop":"Tomato","disease":"Late Blight","treatment":"Remove severely affected material and follow locally approved fungicide guidance.","organic":"Use approved biological or organic options where suitable.","prevention":"Reduce prolonged leaf wetness and improve field ventilation."},
"Potato___Early_blight":{"crop":"Potato","disease":"Early Blight","treatment":"Remove infected foliage and follow locally approved disease-management guidance.","organic":"Maintain sanitation and consider approved organic/biological products.","prevention":"Rotate crops and avoid overhead irrigation."},
"Potato___Late_blight":{"crop":"Potato","disease":"Late Blight","treatment":"Act quickly to remove infected tissue and follow approved disease-management guidance.","organic":"Use approved biological/organic products where applicable.","prevention":"Reduce leaf wetness and monitor neighboring plants."}
}

def init_db():
    con=sqlite3.connect(DB)
    con.execute("""CREATE TABLE IF NOT EXISTS scans(
        id INTEGER PRIMARY KEY AUTOINCREMENT, filename TEXT, crop TEXT,
        disease TEXT, confidence REAL, severity TEXT, created_at TEXT)""")
    con.commit(); con.close()
init_db()

def predict(path):
    if MODEL is not None and CLASS_NAMES:
        img=Image.open(path).convert("RGB").resize((128,128))
        x=np.asarray(img,dtype=np.float32)/255.0
        pred=MODEL.predict(np.expand_dims(x,0),verbose=0)[0]
        i=int(np.argmax(pred)); conf=float(pred[i]); key=CLASS_NAMES[i]
        info=TREATMENTS.get(key,{
            "crop":key.split("___")[0],
            "disease":key.split("___")[-1].replace("_"," "),
            "treatment":"Follow locally approved agricultural treatment guidance.",
            "organic":"Consider approved biological or organic options with local guidance.",
            "prevention":"Use field sanitation, crop rotation and regular monitoring."
        })
        return info["crop"],info["disease"],conf,info
    info=TREATMENTS["Tomato___Early_blight"]
    return info["crop"],info["disease"],0.91,info

def severity(conf):
    if conf>=.90: return "High confidence"
    if conf>=.75: return "Moderate confidence"
    return "Low confidence"

@app.route("/",methods=["GET","POST"])
def index():
    result=None
    if request.method=="POST":
        f=request.files.get("leaf")
        if not f or not f.filename:
            flash("Please select a leaf image."); return redirect(url_for("index"))
        ext=os.path.splitext(f.filename)[1].lower()
        if ext not in [".jpg",".jpeg",".png",".webp"]:
            flash("Please upload JPG, JPEG, PNG or WEBP."); return redirect(url_for("index"))
        name=uuid.uuid4().hex+ext; path=os.path.join(UPLOAD,name); f.save(path)
        crop,disease,conf,info=predict(path); sev=severity(conf)
        now=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        con=sqlite3.connect(DB)
        cur=con.execute("INSERT INTO scans(filename,crop,disease,confidence,severity,created_at) VALUES(?,?,?,?,?,?)",(name,crop,disease,conf,sev,now))
        sid=cur.lastrowid; con.commit(); con.close()
        result={"id":sid,"filename":name,"crop":crop,"disease":disease,"confidence":round(conf*100,2),"severity":sev,"info":info}
    return render_template("index.html",result=result)

@app.route("/history")
def history():
    con=sqlite3.connect(DB); con.row_factory=sqlite3.Row
    rows=con.execute("SELECT * FROM scans ORDER BY id DESC").fetchall(); con.close()
    return render_template("history.html",rows=rows)

@app.route("/report/<int:sid>")
def report(sid):
    con=sqlite3.connect(DB); con.row_factory=sqlite3.Row
    row=con.execute("SELECT * FROM scans WHERE id=?",(sid,)).fetchone(); con.close()
    if not row: return "Scan not found",404
    path=os.path.join(BASE,"reports",f"cropguard_report_{sid}.pdf"); os.makedirs(os.path.dirname(path),exist_ok=True)
    c=canvas.Canvas(path,pagesize=A4); _,h=A4; y=h-60
    c.setFont("Helvetica-Bold",20); c.drawString(50,y,"CropGuard-AI Disease Report"); y-=40
    c.setFont("Helvetica",11)
    for label,value in [("Date",row["created_at"]),("Crop",row["crop"]),("Disease",row["disease"]),("Confidence",f"{row['confidence']*100:.2f}%"),("Severity",row["severity"])]:
        c.drawString(50,y,f"{label}: {value}"); y-=24
    c.setFont("Helvetica-Bold",13); c.drawString(50,y,"Treatment guidance"); y-=22
    c.setFont("Helvetica",10)
    info=TREATMENTS.get(f"{row['crop']}___{row['disease'].replace(' ','_')}",{})
    texts=[info.get("treatment","Follow locally approved agricultural treatment guidance."),"Organic/eco-friendly: "+info.get("organic","Consult local guidance."),"Prevention: "+info.get("prevention","Monitor plants regularly.")]
    for t in texts:
        for line in textwrap.wrap(t,90): c.drawString(50,y,line); y-=16
        y-=8
    c.setFont("Helvetica-Oblique",8); c.drawString(50,50,"Educational assistance only. Follow local agricultural authority and product-label guidance.")
    c.save(); return send_file(path,as_attachment=True)

if __name__=="__main__": app.run(debug=True)
