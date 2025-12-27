import streamlit as st
import os, time
from AI_helper import reset_ai_conversation

def setup_page():
    st.set_page_config(layout="wide")

def show_splash():
    if "app_loaded" not in st.session_state:
        st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
            html, body {margin:0; padding:0; width:100%; height:100%; font-family:'Inter', sans-serif; overflow:hidden;}
            .splash {position:fixed; inset:0; display:flex; flex-direction:column; align-items:center; justify-content:center;
                     background: radial-gradient(circle at top, #6366f1, #0f172a 70%); color:#e5e7eb; z-index:9999;
                     transition:opacity 1s ease, transform 1s ease;}
            .splash.fade-out {opacity:0; transform:scale(1.02); pointer-events:none;}
            canvas {position:fixed; inset:0; z-index:1;}
            .splash > * {position:relative; z-index:2;}
            .logo {font-size:3.2rem; margin-bottom:16px; opacity:0; animation: fadeUp 1s ease forwards;}
            .title {font-size:2.6rem; font-weight:700; letter-spacing:-0.02em; margin-bottom:10px; opacity:0;
                    animation: fadeUp 1s ease forwards; animation-delay:0.4s;}
            .subtitle {font-size:1.15rem; color:#cbd5f5; max-width:560px; text-align:center; line-height:1.5; opacity:0;
                       animation: fadeUp 1s ease forwards; animation-delay:0.8s;}
            .progress {margin-top:32px; width:260px; height:6px; background:rgba(255,255,255,0.15); border-radius:999px;
                       overflow:hidden; opacity:0; animation: fadeUp 0.9s ease forwards; animation-delay:1.4s;}
            .progress-bar {height:100%; width:0%; background:linear-gradient(90deg,#22d3ee,#a78bfa); transition:width 0.3s ease;}
            .footer {position:absolute; bottom:18px; font-size:0.8rem; color:#94a3b8; opacity:0;
                     animation: fadeUp 0.9s ease forwards; animation-delay:2s;}
            @keyframes fadeUp {from{opacity:0; transform:translateY(14px);} to{opacity:1; transform:translateY(0);}}
        </style>

        <canvas id="bg"></canvas>
        <div class="splash" id="splash">
            <div class="logo">📟</div>
            <div class="title">EduData Analytics</div>
            <div class="subtitle">Только данные, которые реально что-то меняют.</div>
            <div class="progress"><div class="progress-bar" id="bar"></div></div>
            <div class="footer">© EduData Analytics • Душанбе</div>
        </div>

        <script>
            const canvas = document.getElementById("bg"), ctx = canvas.getContext("2d");
            canvas.width = innerWidth; canvas.height = innerHeight;
            const particles = Array.from({length:90}, () => ({x:Math.random()*canvas.width,y:Math.random()*canvas.height,
                r:Math.random()*1.5+0.5, vy:Math.random()*0.35+0.1}));
            function animate(){ctx.clearRect(0,0,canvas.width,canvas.height); ctx.fillStyle="rgba(255,255,255,0.75)";
                particles.forEach(p=>{p.y+=p.vy; if(p.y>canvas.height) p.y=0; ctx.beginPath(); ctx.arc(p.x,p.y,p.r,0,Math.PI*2); ctx.fill();});
                requestAnimationFrame(animate);} animate();
            let progress=0, bar=document.getElementById("bar");
            const interval=setInterval(()=>{progress+=Math.random()*16; bar.style.width=Math.min(progress,100)+"%";
                if(progress>=100){clearInterval(interval); setTimeout(()=>{document.getElementById("splash").classList.add("fade-out");},500);}},360);
        </script>
        """, unsafe_allow_html=True)

        import time
        time.sleep(4)
        st.session_state.app_loaded = True
        st.rerun()


def init_api_key():
    if "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

def init_session():
    if "_ai_session_inited" not in st.session_state:
        reset_ai_conversation()
        st.session_state["_ai_session_inited"] = True

def init_page_state():
    if 'page' not in st.session_state:
        st.session_state['page'] = 'Загрузка данных'

def setup_sidebar(set_page):
    st.sidebar.header("🔧 Навигация")
    pages = {
        "Загрузка данных": "📥",
        "Автообработка данных": "🛡️",
        "Обработка пропусков": "⚙️",
        "Обработка выбросов": "🚩",
        "Визуальный анализ": "📊",
        "Сводные таблицы": "📟",
        "Сравнение групп": "⚖️",
        "Логистическая регрессия": "📈",
        "CatBoost моделирование": "🐈‍⬛",
        "Разъяснение результатов (с ИИ)": "💬",
            }

    st.markdown("""
        <style>
            div.stButton > button {
                background-color: #f0f2f6;
                color: black;
                border: 1px solid #ccc;
                border-radius: 6px;
            }
            div.stButton > button:hover {
                background-color: #e0f0ff;
                color: #007BFF;
                border: 1px solid #007BFF;
            }
        </style>
    """, unsafe_allow_html=True)

    for name, icon in pages.items():
        st.sidebar.button(f"{icon} {name}", on_click=set_page, args=(name,))

    if st.sidebar.button("🔄 Очистить всё"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
