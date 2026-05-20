import streamlit as st
import os, time
from Utils.AI_helper import reset_ai_memory
from Utils.i18n import t

def setup_page():
    st.set_page_config(layout="wide")

def show_splash():
    if "app_loaded" not in st.session_state:
        subtitle = t("Интеллектуальная система анализа и прогнозирования образовательных данных")
        ministry = t("Разработано для Министерства образования и науки РТ")
        splash_html = """
        <style>
            [data-testid="stSidebar"] {display: none;}
            @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;800&display=swap');
            html, body {margin:0; padding:0; width:100%; height:100%; font-family:'Outfit', sans-serif; overflow:hidden;}
            .splash {position:fixed; inset:0; display:flex; flex-direction:column; align-items:center; justify-content:center;
                     background: radial-gradient(circle at center, #1e293b, #0f172a 90%); color:#f8fafc; z-index:9999;
                     transition:opacity 0.8s ease, transform 0.8s ease;}
            .splash.fade-out {opacity:0; transform:scale(1.05); pointer-events:none;}
            canvas {position:fixed; inset:0; z-index:1; opacity: 0.6;}
            .splash > * {position:relative; z-index:2;}
            
            .logo-icon {font-size:4rem; margin-bottom:10px; background: linear-gradient(135deg, #6366f1, #a855f7); 
                        -webkit-background-clip: text; -webkit-text-fill-color: transparent; 
                        filter: drop-shadow(0 0 20px rgba(99, 102, 241, 0.5));
                        animation: float 3s ease-in-out infinite;}
            
            .title {font-size:3.5rem; font-weight:800; letter-spacing:-0.03em; margin-bottom:8px; opacity:0;
                    background: linear-gradient(to right, #fff, #cbd5e1); -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                    animation: fadeUp 0.8s ease forwards; animation-delay:0.3s;}
            
            .subtitle {font-size:1.2rem; color:#94a3b8; font-weight:400; opacity:0; letter-spacing: 0.02em; text-align: center; max-width: 800px;
                       animation: fadeUp 0.8s ease forwards; animation-delay:0.6s;}
            
            .ministry {font-size:0.9rem; color:#64748b; margin-top: 15px; opacity:0;
                       animation: fadeUp 0.8s ease forwards; animation-delay:0.9s;}

            .loader-line {margin-top:40px; width:200px; height:4px; background:rgba(255,255,255,0.1); border-radius:4px; overflow:hidden;
                          opacity:0; animation: fadeUp 0.8s ease forwards; animation-delay:1.2s;}
            .loader-fill {height:100%; width:0%; background: #6366f1; box-shadow: 0 0 10px #6366f1; transition:width 0.2s linear;}
            
            @keyframes fadeUp {from{opacity:0; transform:translateY(20px);} to{opacity:1; transform:translateY(0);}}
            @keyframes float {0%, 100%{transform:translateY(0);} 50%{transform:translateY(-10px);}}
        </style>

        <canvas id="bits"></canvas>
        <div class="splash" id="splash">
            <div class="logo-icon">❖</div>
            <div class="title">EduStat AI</div>
            <div class="subtitle">__SPLASH_SUBTITLE__</div>
            <div class="ministry">__SPLASH_MINISTRY__</div>
            <div class="loader-line"><div class="loader-fill" id="fill"></div></div>
        </div>

        <script>
            const canvas = document.getElementById("bits"), ctx = canvas.getContext("2d");
            let w, h, particles = [];
            
            function resize() { w = canvas.width = innerWidth; h = canvas.height = innerHeight; }
            window.addEventListener('resize', resize); resize();

            // Создаем частицы "Нейросеть"
            for(let i=0; i<60; i++) {
                particles.push({
                    x: Math.random()*w, y: Math.random()*h,
                    vx: (Math.random()-0.5)*0.5, vy: (Math.random()-0.5)*0.5,
                    size: Math.random()*2 + 1
                });
            }

            function animate() {
                ctx.clearRect(0,0,w,h);
                ctx.fillStyle = "#6366f1";
                ctx.strokeStyle = "rgba(99, 102, 241, 0.15)";
                
                for(let i=0; i<particles.length; i++) {
                    let p = particles[i];
                    p.x += p.vx; p.y += p.vy;
                    
                    if(p.x < 0 || p.x > w) p.vx *= -1;
                    if(p.y < 0 || p.y > h) p.vy *= -1;
                    
                    ctx.beginPath();
                    ctx.arc(p.x, p.y, p.size, 0, Math.PI*2);
                    ctx.fill();
                    
                    // Соединяем линии
                    for(let j=i+1; j<particles.length; j++) {
                        let p2 = particles[j];
                        let dist = Math.hypot(p.x-p2.x, p.y-p2.y);
                        if(dist < 150) {
                            ctx.lineWidth = 1 - dist/150;
                            ctx.beginPath();
                            ctx.moveTo(p.x, p.y);
                            ctx.lineTo(p2.x, p2.y);
                            ctx.stroke();
                        }
                    }
                }
                requestAnimationFrame(animate);
            }
            animate();

            // Прогресс бар
            let progress = 0;
            const bar = document.getElementById("fill");
            const timer = setInterval(() => {
                progress += Math.random() * 8;
                if(progress > 100) progress = 100;
                bar.style.width = progress + "%";
                if(progress === 100) {
                    clearInterval(timer);
                    setTimeout(() => {
                        document.getElementById("splash").classList.add("fade-out");
                    }, 600);
                }
            }, 200);
        </script>
        """
        splash_html = (
            splash_html
            .replace("__SPLASH_SUBTITLE__", subtitle)
            .replace("__SPLASH_MINISTRY__", ministry)
        )
        st.markdown(splash_html, unsafe_allow_html=True)

        import time
        time.sleep(4)
        st.session_state.app_loaded = True
        st.rerun()


def init_api_key():
    if "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

def init_session():
    if "_ai_session_inited" not in st.session_state:
        reset_ai_memory()
        st.session_state["_ai_session_inited"] = True

def init_page_state():
    if 'page' not in st.session_state:
        st.session_state['page'] = t('Загрузка данных')

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
                border-radius: 6px;
            }
        </style>
    """, unsafe_allow_html=True)

    for name, icon in pages.items():
        st.sidebar.button(f"{icon} {name}", on_click=set_page, args=(name,))

    if st.sidebar.button("🔄 Очистить всё"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
