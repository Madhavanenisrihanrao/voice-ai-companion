/* =============================================================
   AI Companion — Frontend Application
   =============================================================
   Three.js animated sphere · Socket.IO real-time chat
   Web Speech API for TTS / STT
   ============================================================= */

import * as THREE from 'three';

// ── Animation Constants ──────────────────────────────────────
const IDLE_AMP    = 0.12;
const SPEAK_AMP   = 0.42;
const LISTEN_AMP  = 0.22;
const THINK_AMP   = 0.28;
const LERP_SPEED  = 0.04;

// ── Emotion → Sphere Colour Palette ─────────────────────────
const EMOTION_PALETTE = {
    happy:    { c1: '#fbbf24', c2: '#f59e0b', c3: '#f97316', glow: '#fbbf24' },
    sad:      { c1: '#3b82f6', c2: '#6366f1', c3: '#1d4ed8', glow: '#3b82f6' },
    angry:    { c1: '#ef4444', c2: '#dc2626', c3: '#f97316', glow: '#ef4444' },
    neutral:  { c1: '#8b5cf6', c2: '#6366f1', c3: '#06b6d4', glow: '#8b5cf6' },
    surprise: { c1: '#06b6d4', c2: '#22d3ee', c3: '#8b5cf6', glow: '#06b6d4' },
    fear:     { c1: '#a855f7', c2: '#7c3aed', c3: '#c026d3', glow: '#a855f7' },
    disgust:  { c1: '#84cc16', c2: '#65a30d', c3: '#eab308', glow: '#84cc16' },
};

const EMOTION_ICONS = {
    happy: '😊', sad: '😢', angry: '😠', surprise: '😮',
    fear: '😨', disgust: '🤢', neutral: '😐',
};

// ── Three.js Globals ─────────────────────────────────────────
let scene, camera, renderer, clock;
let mainSphere, glowSphere, particleSystem;
let targetAmp   = IDLE_AMP;
let currentAmp  = IDLE_AMP;
let targetColors = EMOTION_PALETTE.neutral;

// ── App State ────────────────────────────────────────────────
let socket;
let isSpeaking  = false;
let isListening = false;
let isWaitingForResponse = false;  // True when waiting for AI to respond
let currentEmotion = 'neutral';
let currentLang    = 'en';      // BCP-47 primary subtag from server
let recognition;
let sessionEnded = false;
let selfStream   = null;        // getUserMedia stream

// Language code → BCP-47 for TTS / STT
const LANG_MAP = {
    'en': 'en-US',  'hi': 'hi-IN',  'te': 'te-IN',
    'ta': 'ta-IN',  'kn': 'kn-IN',  'ml': 'ml-IN',
    'mr': 'mr-IN',  'bn': 'bn-IN',  'gu': 'gu-IN',
    'pa': 'pa-IN',  'ur': 'ur-PK',  'es': 'es-ES',
    'fr': 'fr-FR',  'de': 'de-DE',  'ja': 'ja-JP',
    'ko': 'ko-KR',  'zh': 'zh-CN',  'ar': 'ar-SA',
    'pt': 'pt-BR',  'ru': 'ru-RU',  'it': 'it-IT',
};

const LANG_NAMES = {
    'en': 'English', 'hi': 'Hindi',  'te': 'Telugu',
    'ta': 'Tamil',   'kn': 'Kannada','ml': 'Malayalam',
    'mr': 'Marathi', 'bn': 'Bengali','gu': 'Gujarati',
    'pa': 'Punjabi', 'ur': 'Urdu',   'es': 'Spanish',
    'fr': 'French',  'de': 'German', 'ja': 'Japanese',
    'ko': 'Korean',  'zh': 'Chinese','ar': 'Arabic',
    'pt': 'Portuguese','ru':'Russian','it': 'Italian',
};

// ── DOM Refs (populated in cacheDom) ─────────────────────────
let messagesEl, inputEl, sendBtn, micBtn;
let emotionIcon, emotionText, statusDot, statusText;
let langIcon, langText;

// =============================================================
//  BOOT
// =============================================================
document.addEventListener('DOMContentLoaded', () => {
    cacheDom();
    initThreeJS();
    initSocket();
    initSpeechRecognition();
    initSelfView();
    initUI();
    animate();
});

function cacheDom() {
    messagesEl  = document.getElementById('messages');
    inputEl     = document.getElementById('user-input');
    sendBtn     = document.getElementById('send-btn');
    micBtn      = document.getElementById('mic-btn');
    emotionIcon = document.getElementById('emotion-icon');
    emotionText = document.getElementById('emotion-text');
    statusDot   = document.getElementById('status-dot');
    statusText  = document.getElementById('status-text');
    langIcon    = document.getElementById('lang-icon');
    langText    = document.getElementById('lang-text');
}

// =============================================================
//  THREE.JS SCENE
// =============================================================

// ── Vertex shader — simplex noise displacement ───────────────
const vertexShader = `
uniform float uTime;
uniform float uAmplitude;
uniform float uFrequency;
varying vec3  vNormal;
varying vec3  vPosition;
varying float vDisplacement;

/* --- 3-D Simplex Noise (Ashima Arts / Stefan Gustavson) --- */
vec3 mod289(vec3 x){return x-floor(x*(1./289.))*289.;}
vec4 mod289(vec4 x){return x-floor(x*(1./289.))*289.;}
vec4 permute(vec4 x){return mod289(((x*34.)+10.)*x);}
vec4 taylorInvSqrt(vec4 r){return 1.79284291400159-.85373472095314*r;}

float snoise(vec3 v){
    const vec2 C=vec2(1./6.,1./3.);
    const vec4 D=vec4(0.,.5,1.,2.);
    vec3 i=floor(v+dot(v,C.yyy));
    vec3 x0=v-i+dot(i,C.xxx);
    vec3 g=step(x0.yzx,x0.xyz);
    vec3 l=1.-g;
    vec3 i1=min(g,l.zxy);
    vec3 i2=max(g,l.zxy);
    vec3 x1=x0-i1+C.xxx;
    vec3 x2=x0-i2+C.yyy;
    vec3 x3=x0-D.yyy;
    i=mod289(i);
    vec4 p=permute(permute(permute(
        i.z+vec4(0.,i1.z,i2.z,1.))
       +i.y+vec4(0.,i1.y,i2.y,1.))
       +i.x+vec4(0.,i1.x,i2.x,1.));
    float n_=.142857142857;
    vec3 ns=n_*D.wyz-D.xzx;
    vec4 j=p-49.*floor(p*ns.z*ns.z);
    vec4 x_=floor(j*ns.z);
    vec4 y_=floor(j-7.*x_);
    vec4 xx=x_*ns.x+ns.yyyy;
    vec4 yy=y_*ns.x+ns.yyyy;
    vec4 h=1.-abs(xx)-abs(yy);
    vec4 b0=vec4(xx.xy,yy.xy);
    vec4 b1=vec4(xx.zw,yy.zw);
    vec4 s0=floor(b0)*2.+1.;
    vec4 s1=floor(b1)*2.+1.;
    vec4 sh=-step(h,vec4(0.));
    vec4 a0=b0.xzyw+s0.xzyw*sh.xxyy;
    vec4 a1=b1.xzyw+s1.xzyw*sh.zzww;
    vec3 p0=vec3(a0.xy,h.x);
    vec3 p1=vec3(a0.zw,h.y);
    vec3 p2=vec3(a1.xy,h.z);
    vec3 p3=vec3(a1.zw,h.w);
    vec4 norm=taylorInvSqrt(vec4(dot(p0,p0),dot(p1,p1),dot(p2,p2),dot(p3,p3)));
    p0*=norm.x;p1*=norm.y;p2*=norm.z;p3*=norm.w;
    vec4 m=max(.6-vec4(dot(x0,x0),dot(x1,x1),dot(x2,x2),dot(x3,x3)),0.);
    m=m*m;
    return 42.*dot(m*m,vec4(dot(p0,x0),dot(p1,x1),dot(p2,x2),dot(p3,x3)));
}

void main(){
    float n1 = snoise(normal * uFrequency + uTime * 0.35);
    float n2 = snoise(normal * uFrequency * 2.0 + uTime * 0.55) * 0.5;
    float displacement = (n1 + n2) * uAmplitude;
    vec3 newPos = position + normal * displacement;

    vNormal       = normalize(normalMatrix * normal);
    vPosition     = (modelViewMatrix * vec4(newPos, 1.0)).xyz;
    vDisplacement = displacement;

    gl_Position = projectionMatrix * modelViewMatrix * vec4(newPos, 1.0);
}
`;

// ── Fragment shader — iridescent glass ───────────────────────
const fragmentShader = `
uniform float uTime;
uniform vec3  uColor1;
uniform vec3  uColor2;
uniform vec3  uColor3;
uniform float uOpacity;
varying vec3  vNormal;
varying vec3  vPosition;
varying float vDisplacement;

void main(){
    vec3 viewDir = normalize(-vPosition);
    float fresnel = pow(1.0 - max(dot(viewDir, vNormal), 0.0), 3.0);

    vec3 base = mix(uColor1, uColor2, smoothstep(-0.3, 0.3, vDisplacement));
    base = mix(base, uColor3, fresnel);
    base += 0.06 * sin(uTime * 2.0 + vPosition.y * 4.0);

    float alpha = uOpacity + fresnel * 0.35;
    gl_FragColor = vec4(base, alpha);
}
`;

// ── Glow shaders ─────────────────────────────────────────────
const glowVert = `
varying vec3 vNormal;
void main(){
    vNormal = normalize(normalMatrix * normal);
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}`;
const glowFrag = `
uniform vec3  uGlowColor;
uniform float uIntensity;
varying vec3  vNormal;
void main(){
    float rim = pow(0.6 - dot(vNormal, vec3(0.0, 0.0, 1.0)), 2.0);
    gl_FragColor = vec4(uGlowColor, rim * uIntensity);
}`;

function initThreeJS() {
    scene  = new THREE.Scene();
    camera = new THREE.PerspectiveCamera(
        75, window.innerWidth / window.innerHeight, 0.1, 100
    );
    camera.position.z = 4;

    renderer = new THREE.WebGLRenderer({
        canvas: document.getElementById('sphere-canvas'),
        antialias: true,
        alpha: true,
    });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

    clock = new THREE.Clock();

    createMainSphere();
    createGlow();
    createParticles();

    scene.add(new THREE.AmbientLight(0x404060, 0.5));
    const dir = new THREE.DirectionalLight(0x8888ff, 0.3);
    dir.position.set(2, 3, 5);
    scene.add(dir);

    window.addEventListener('resize', () => {
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(window.innerWidth, window.innerHeight);
    });
}

function createMainSphere() {
    const geo = new THREE.IcosahedronGeometry(1.5, 64);
    const mat = new THREE.ShaderMaterial({
        uniforms: {
            uTime:      { value: 0 },
            uAmplitude: { value: IDLE_AMP },
            uFrequency: { value: 1.5 },
            uColor1:    { value: new THREE.Color(EMOTION_PALETTE.neutral.c1) },
            uColor2:    { value: new THREE.Color(EMOTION_PALETTE.neutral.c2) },
            uColor3:    { value: new THREE.Color(EMOTION_PALETTE.neutral.c3) },
            uOpacity:   { value: 0.7 },
        },
        vertexShader,
        fragmentShader,
        transparent: true,
        side: THREE.DoubleSide,
        depthWrite: false,
    });
    mainSphere = new THREE.Mesh(geo, mat);
    scene.add(mainSphere);
}

function createGlow() {
    const geo = new THREE.IcosahedronGeometry(1.9, 32);
    const mat = new THREE.ShaderMaterial({
        uniforms: {
            uGlowColor: { value: new THREE.Color(EMOTION_PALETTE.neutral.glow) },
            uIntensity: { value: 0.2 },
        },
        vertexShader: glowVert,
        fragmentShader: glowFrag,
        transparent: true,
        blending: THREE.AdditiveBlending,
        side: THREE.BackSide,
        depthWrite: false,
    });
    glowSphere = new THREE.Mesh(geo, mat);
    scene.add(glowSphere);
}

function createParticles() {
    const count = 300;
    const pos   = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
        const theta = Math.random() * Math.PI * 2;
        const phi   = Math.acos(2 * Math.random() - 1);
        const r     = 2.2 + Math.random() * 2.0;
        pos[i * 3]     = r * Math.sin(phi) * Math.cos(theta);
        pos[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta);
        pos[i * 3 + 2] = r * Math.cos(phi);
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    const mat = new THREE.PointsMaterial({
        color: 0x8b5cf6,
        size: 0.018,
        transparent: true,
        opacity: 0.45,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
    });
    particleSystem = new THREE.Points(geo, mat);
    scene.add(particleSystem);
}

// =============================================================
//  ANIMATION LOOP
// =============================================================
function animate() {
    requestAnimationFrame(animate);

    const t = clock.getElapsedTime();

    // Smooth amplitude interpolation
    currentAmp += (targetAmp - currentAmp) * LERP_SPEED;

    // Sphere uniforms
    const u = mainSphere.material.uniforms;
    u.uTime.value      = t;
    u.uAmplitude.value  = currentAmp;

    // Gentle rotation
    mainSphere.rotation.y = t * 0.08;
    mainSphere.rotation.x = Math.sin(t * 0.04) * 0.12;

    // Glow follows sphere and scales with amplitude
    glowSphere.rotation.copy(mainSphere.rotation);
    glowSphere.material.uniforms.uIntensity.value = 0.15 + currentAmp * 0.6;

    // Particles drift
    particleSystem.rotation.y = t * 0.025;
    particleSystem.rotation.x = Math.sin(t * 0.015) * 0.08;

    // Smoothly lerp colours toward target emotion palette
    const tc1  = new THREE.Color(targetColors.c1);
    const tc2  = new THREE.Color(targetColors.c2);
    const tc3  = new THREE.Color(targetColors.c3);
    const tg   = new THREE.Color(targetColors.glow);
    u.uColor1.value.lerp(tc1, 0.02);
    u.uColor2.value.lerp(tc2, 0.02);
    u.uColor3.value.lerp(tc3, 0.02);
    glowSphere.material.uniforms.uGlowColor.value.lerp(tg, 0.02);

    renderer.render(scene, camera);
}

// =============================================================
//  EMOTION
// =============================================================
function setEmotion(emotion, confidence) {
    if (confidence === undefined) {
        confidence = 1.0;
    }

    const confPercent = Math.round(confidence * 100);

    if (confidence > 0.70) {
        currentEmotion = emotion;
        targetColors   = EMOTION_PALETTE[emotion] || EMOTION_PALETTE.neutral;
        if (emotionIcon) emotionIcon.textContent = EMOTION_ICONS[emotion] || '😐';
        if (emotionText) emotionText.textContent  = `Detected facial expression: ${emotion.charAt(0).toUpperCase() + emotion.slice(1)} (${confPercent}%)`;
    } else if (confidence >= 0.40) {
        currentEmotion = emotion;
        targetColors   = EMOTION_PALETTE[emotion] || EMOTION_PALETTE.neutral;
        if (emotionIcon) emotionIcon.textContent = EMOTION_ICONS[emotion] || '😐';
        if (emotionText) emotionText.textContent  = `Possible facial expression: ${emotion.charAt(0).toUpperCase() + emotion.slice(1)} (${confPercent}%)`;
    } else {
        currentEmotion = 'neutral';
        targetColors   = EMOTION_PALETTE.neutral;
        if (emotionIcon) emotionIcon.textContent = '😐';
        if (emotionText) emotionText.textContent  = "Facial expression uncertain.";
    }
}

// =============================================================
//  SOCKET.IO
// =============================================================
function initSocket() {
    socket = io();

    socket.on('connect', () => {
        statusDot.className  = 'connected';
        statusText.textContent = 'Connected';
    });

    socket.on('disconnect', () => {
        statusDot.className  = 'disconnected';
        statusText.textContent = 'Disconnected';
    });

    socket.on('ai_response', (data) => {
        isWaitingForResponse = false;
        hideTyping();
        addMessage('ai', data.text);
        if (data.emotion) setEmotion(data.emotion, data.confidence);
        if (data.lang) setLanguage(data.lang);
        speakText(data.text, data.lang);
        if (data.farewell) disableInput();
    });

    socket.on('ai_proactive', (data) => {
        // AI is proactively commenting on what it sees
        isWaitingForResponse = false;
        addMessage('ai', data.text);
        if (data.emotion) setEmotion(data.emotion, data.confidence);
        if (data.lang) setLanguage(data.lang);
        speakText(data.text, data.lang);
    });

    socket.on('emotion_update', (data) => {
        if (data.emotion) setEmotion(data.emotion, data.confidence);
    });

    socket.on('session_ended', () => {
        addMessage('system', 'Session saved. Goodbye!');
        disableInput();
    });

    socket.on('error', (data) => {
        addMessage('system', data.message || 'An error occurred.');
    });
}

// Vision keywords that trigger image analysis
const VISION_KEYWORDS = [
    'look at this', 'what is this', 'what do you see', 'can you see',
    'show you', 'showing you', 'look here', 'see this', 'what am i holding',
    'what is in my hand', 'read this', 'what does this say', 'describe this',
    'చూడు', 'ఇది ఏంటి', 'చదువు', 'देखो', 'यह क्या है', 'पढ़ो',
];

function sendMessage(text) {
    text = (text || '').trim();
    if (!text || sessionEnded) return;
    
    // Check if user is asking about something visual
    const lowerText = text.toLowerCase();
    const isVisionQuery = VISION_KEYWORDS.some(kw => lowerText.includes(kw));
    
    addMessage('user', text);
    showTyping();
    isWaitingForResponse = true;
    targetAmp = THINK_AMP;
    
    if (isVisionQuery && selfStream) {
        // Capture current frame and send for vision analysis
        sendVisionAnalysis(text);
    } else {
        socket.emit('user_message', { text });
    }
    
    inputEl.value = '';
    inputEl.focus();
}

function sendVisionAnalysis(question) {
    const video = document.getElementById('self-video');
    if (!video || !selfStream) {
        socket.emit('user_message', { text: question });
        return;
    }
    
    const canvas = document.createElement('canvas');
    canvas.width = 640;
    canvas.height = 480;
    const ctx2d = canvas.getContext('2d');
    ctx2d.drawImage(video, 0, 0, 640, 480);
    const dataUrl = canvas.toDataURL('image/jpeg', 0.8);
    
    socket.emit('analyze_image', { image: dataUrl, question: question });
}

// =============================================================
//  TTS — Server-side gTTS via Audio element (reliable playback)
// =============================================================
let ttsAudio = null;

function speakText(text, lang) {
    if (!text) return;

    // Stop anything currently playing
    if (ttsAudio) {
        ttsAudio.pause();
        ttsAudio.src = '';
        ttsAudio = null;
    }
    speechSynthesis.cancel();

    const ttsLang = lang || currentLang || 'en';
    console.log(`[TTS] Speaking in ${ttsLang}: "${text.substring(0, 50)}..."`);

    // Check if text contains non-Latin script (Telugu, Hindi, Arabic, etc.)
    // These need gTTS even if language was detected as 'en'
    const hasNonLatinScript = /[\u0900-\u0D7F\u0600-\u06FF\u0400-\u04FF\u3040-\u9FFF\uAC00-\uD7AF]/.test(text);
    if (hasNonLatinScript) {
        console.log('[TTS] Non-Latin script detected, forcing gTTS');
    }

    // For English with Latin script only: use browser-native TTS (faster, lower latency)
    if (ttsLang === 'en' && !hasNonLatinScript && 'speechSynthesis' in window) {
        const utt = new SpeechSynthesisUtterance(text);
        utt.rate   = 1.0;
        utt.pitch  = 1.0;
        utt.volume = 1.0;
        utt.lang   = 'en-US';

        const voices = speechSynthesis.getVoices();
        const pref = voices.find(v => v.name.includes('Zira'))
                  || voices.find(v => /Google.*US/i.test(v.name))
                  || voices.find(v => v.lang.startsWith('en'));
        if (pref) utt.voice = pref;

        utt.onstart = () => { console.log('[TTS] Browser TTS started'); isSpeaking = true;  targetAmp = SPEAK_AMP; };
        utt.onend   = () => { console.log('[TTS] Browser TTS ended'); isSpeaking = false; targetAmp = IDLE_AMP;  };
        utt.onerror = (e) => { console.warn('[TTS] Browser TTS error:', e); isSpeaking = false; targetAmp = IDLE_AMP;  };

        speechSynthesis.speak(utt);
        return;
    }

    // For non-English: fetch gTTS audio as blob and play via Audio element
    console.log(`[TTS] Fetching gTTS audio for ${ttsLang}...`);
    isSpeaking = true;
    targetAmp  = SPEAK_AMP;

    const url = `/tts?text=${encodeURIComponent(text)}&lang=${encodeURIComponent(ttsLang)}`;
    fetch(url)
        .then(r => {
            console.log(`[TTS] gTTS response: ${r.status}, type: ${r.headers.get('content-type')}`);
            if (!r.ok) throw new Error(`TTS fetch failed: ${r.status}`);
            return r.blob();
        })
        .then(blob => {
            console.log(`[TTS] Got audio blob: ${blob.size} bytes, type: ${blob.type}`);
            const blobUrl = URL.createObjectURL(blob);
            ttsAudio = new Audio(blobUrl);
            ttsAudio.volume = 1.0;
            
            ttsAudio.onloadeddata = () => console.log('[TTS] Audio loaded');
            ttsAudio.onplay = () => console.log('[TTS] Audio playing');
            ttsAudio.onended = () => {
                console.log('[TTS] Audio ended');
                isSpeaking = false;
                targetAmp = IDLE_AMP;
                URL.revokeObjectURL(blobUrl);
                ttsAudio = null;
            };
            ttsAudio.onerror = (e) => {
                console.error('[TTS] Audio error:', e, ttsAudio?.error);
                isSpeaking = false;
                targetAmp = IDLE_AMP;
                URL.revokeObjectURL(blobUrl);
                ttsAudio = null;
            };
            
            return ttsAudio.play();
        })
        .then(() => {
            console.log('[TTS] Play promise resolved');
        })
        .catch(err => {
            console.error('[TTS] gTTS playback failed:', err);
            isSpeaking = false;
            targetAmp  = IDLE_AMP;
        });
}

// Voices often load async — warm up the list
if ('speechSynthesis' in window) {
    speechSynthesis.getVoices();
    speechSynthesis.onvoiceschanged = () => speechSynthesis.getVoices();
}

// =============================================================
//  WEB SPEECH API — STT
// =============================================================
function initSpeechRecognition() {
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SR) {
        if (micBtn) micBtn.style.display = 'none';
        return;
    }
    recognition = new SR();
    recognition.continuous      = false;
    recognition.interimResults  = false;
    recognition.lang            = 'en-US';  // will be updated dynamically

    recognition.onresult = (e) => {
        const transcript = e.results[0][0].transcript;
        sendMessage(transcript);
    };
    recognition.onerror = () => stopListening();
    recognition.onend   = () => stopListening();
}

function toggleMic() {
    if (!recognition || sessionEnded) return;
    if (isListening) {
        recognition.stop();
    } else {
        speechSynthesis.cancel();
        isSpeaking = false;
        // Set recognition language to detected language
        recognition.lang = LANG_MAP[currentLang] || 'en-US';
        recognition.start();
        isListening = true;
        micBtn.classList.add('listening');
        targetAmp = LISTEN_AMP;
    }
}

function stopListening() {
    isListening = false;
    if (micBtn) micBtn.classList.remove('listening');
    if (!isSpeaking) targetAmp = IDLE_AMP;
}

// =============================================================
//  UI
// =============================================================
function initUI() {
    document.getElementById('input-bar').addEventListener('submit', (e) => {
        e.preventDefault();
        sendMessage(inputEl.value);
    });

    micBtn.addEventListener('click', toggleMic);

    // Camera toggle (self-view)
    const camToggle = document.getElementById('cam-toggle');
    if (camToggle) {
        camToggle.addEventListener('click', toggleSelfView);
    }

    inputEl.focus();
}

function addMessage(type, text) {
    const msg = document.createElement('div');
    msg.className = `message ${type}`;

    if (type !== 'system') {
        const label = document.createElement('span');
        label.className = 'msg-label' + (type === 'user' ? ' user-label' : '');
        label.textContent = type === 'user' ? 'You' : 'AI Companion';
        msg.appendChild(label);
    }

    const content = document.createElement('span');
    content.className = 'msg-text';
    content.textContent = text;
    msg.appendChild(content);

    messagesEl.appendChild(msg);
    messagesEl.scrollTop = messagesEl.scrollHeight;
}

function showTyping() {
    if (document.getElementById('typing-indicator')) return;
    const el = document.createElement('div');
    el.id = 'typing-indicator';
    el.className = 'message ai typing';
    el.innerHTML = '<span class="dot"></span><span class="dot"></span><span class="dot"></span>';
    messagesEl.appendChild(el);
    messagesEl.scrollTop = messagesEl.scrollHeight;
}

function hideTyping() {
    const el = document.getElementById('typing-indicator');
    if (el) el.remove();
}

function disableInput() {
    sessionEnded = true;
    inputEl.disabled  = true;
    sendBtn.disabled  = true;
    micBtn.disabled   = true;
    inputEl.placeholder = 'Session ended';
}

// =============================================================
//  LANGUAGE
// =============================================================
function setLanguage(lang) {
    if (!lang) return;
    currentLang = lang;
    const name = LANG_NAMES[lang] || lang.toUpperCase();
    if (langIcon) langIcon.textContent = '\uD83C\uDF10';
    if (langText) langText.textContent = name;
}

// =============================================================
//  SELF-VIEW CAMERA (getUserMedia) + Draggable + Emotion Frames
// =============================================================
let emotionInterval = null;

function initSelfView() {
    const video = document.getElementById('self-video');
    const selfView = document.getElementById('self-view');
    if (!video || !selfView) return;

    // ── Make the self-view draggable ─────────────────────────
    makeDraggable(selfView);

    navigator.mediaDevices.getUserMedia({ video: true, audio: false })
        .then((stream) => {
            selfStream = stream;
            video.srcObject = stream;
            selfView.classList.remove('cam-off');
            startBrowserEmotionLoop();
        })
        .catch(() => {
            selfView.classList.add('cam-off');
        });
}

function toggleSelfView() {
    const video = document.getElementById('self-video');
    const selfView = document.getElementById('self-view');
    if (!selfView) return;

    if (selfStream) {
        // Turn off camera
        selfStream.getTracks().forEach(t => t.stop());
        selfStream = null;
        if (video) video.srcObject = null;
        selfView.classList.add('cam-off');
        stopBrowserEmotionLoop();
    } else {
        // Turn on camera
        navigator.mediaDevices.getUserMedia({ video: true, audio: false })
            .then((stream) => {
                selfStream = stream;
                if (video) video.srcObject = stream;
                selfView.classList.remove('cam-off');
                startBrowserEmotionLoop();
            })
            .catch(() => {
                selfView.classList.add('cam-off');
            });
    }
}

// ── Draggable helper ─────────────────────────────────────────
function makeDraggable(el) {
    let isDragging = false, startX, startY, origX, origY;

    const onPointerDown = (e) => {
        // Ignore clicks on buttons inside the element
        if (e.target.closest('button')) return;
        isDragging = true;
        startX = e.clientX;
        startY = e.clientY;
        const rect = el.getBoundingClientRect();
        origX = rect.left;
        origY = rect.top;
        el.style.cursor = 'grabbing';
        el.style.transition = 'none';
        e.preventDefault();
    };

    const onPointerMove = (e) => {
        if (!isDragging) return;
        const dx = e.clientX - startX;
        const dy = e.clientY - startY;
        let newX = origX + dx;
        let newY = origY + dy;
        // Clamp to viewport
        newX = Math.max(0, Math.min(window.innerWidth  - el.offsetWidth,  newX));
        newY = Math.max(0, Math.min(window.innerHeight - el.offsetHeight, newY));
        el.style.left   = newX + 'px';
        el.style.top    = newY + 'px';
        el.style.bottom = 'auto';
        el.style.right  = 'auto';
    };

    const onPointerUp = () => {
        if (!isDragging) return;
        isDragging = false;
        el.style.cursor = 'grab';
        el.style.transition = 'box-shadow 0.3s';
    };

    el.addEventListener('pointerdown', onPointerDown);
    document.addEventListener('pointermove', onPointerMove);
    document.addEventListener('pointerup', onPointerUp);
    el.style.cursor = 'grab';
}

// ── Browser-frame emotion detection (send frames to server) ──
let visionInterval = null;

function startBrowserEmotionLoop() {
    if (emotionInterval || !socket) return;
    const video = document.getElementById('self-video');
    if (!video) return;

    const canvas = document.createElement('canvas');
    const ctx2d  = canvas.getContext('2d');

    // Emotion detection frames (low quality, frequent)
    emotionInterval = setInterval(() => {
        if (!selfStream || !video.videoWidth) return;
        canvas.width  = 320;
        canvas.height = 240;
        ctx2d.drawImage(video, 0, 0, 320, 240);
        const dataUrl = canvas.toDataURL('image/jpeg', 0.5);
        socket.emit('browser_frame', { image: dataUrl });
    }, 2000);   // every 2 seconds
    
    // Proactive vision frames (higher quality, less frequent)
    // Only send when AI is truly idle (not speaking, not waiting for response)
    visionInterval = setInterval(() => {
        if (!selfStream || !video.videoWidth) return;
        if (isSpeaking || isWaitingForResponse) {
            console.log('[Vision] Skipping proactive - AI busy (speaking:', isSpeaking, 'waiting:', isWaitingForResponse, ')');
            return;
        }
        canvas.width  = 640;
        canvas.height = 480;
        ctx2d.drawImage(video, 0, 0, 640, 480);
        const dataUrl = canvas.toDataURL('image/jpeg', 0.7);
        socket.emit('vision_frame', { image: dataUrl });
    }, 15000);   // every 15 seconds for proactive observation
}

function stopBrowserEmotionLoop() {
    if (emotionInterval) {
        clearInterval(emotionInterval);
        emotionInterval = null;
    }
    if (visionInterval) {
        clearInterval(visionInterval);
        visionInterval = null;
    }
}
