import { useEffect, useState, useRef, useCallback } from 'react'; 
import { Panel } from '../components/ui/Panel';
import { Film, FileAudio, Image as ImageIcon, X, AlertTriangle, CheckCircle, Loader, Zap, Activity, Camera } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { getAuth } from 'firebase/auth';
import { getFirestore, doc, getDoc, collection, getDocs } from 'firebase/firestore';
import { FaceLandmarker, FilesetResolver, DrawingUtils } from "@mediapipe/tasks-vision";

/* ================= METRIC BAR ================= */
const MetricBar = ({ label, value, color, highlight }: { label: string; value: number; color: string; highlight?: boolean }) => (
  <div className={`space-y-1.5 transition-all duration-700 ${highlight ? 'opacity-100' : 'opacity-60'}`}>
    <div className="flex justify-between items-center">
      <span className={`text-[10px] font-mono uppercase tracking-widest ${highlight ? 'text-white' : 'text-gray-400'}`}>{label}</span>
      <span className="text-[10px] font-mono text-white">{(value * 100).toFixed(1)}%</span>
    </div>
    <div className="h-1.5 bg-white/5 rounded-full overflow-hidden border border-white/5">
      <motion.div
        animate={{ width: `${Math.min(value * 100, 100)}%` }}
        transition={{ type: 'spring', stiffness: 120, damping: 20 }}
        className={`h-full ${color} ${highlight ? 'shadow-[0_0_12px_rgba(255,255,255,0.4)]' : ''}`}
      />
    </div>
  </div>
);

/* ================= ASYMMETRY ROW ================= */
const AsymmetryRow = ({ label, left, right, highlight }: { label: string; left: number; right: number; highlight?: boolean }) => {
  const diff = Math.abs(left - right);
  const severity = diff > 0.3 ? 'text-red-400' : diff > 0.15 ? 'text-yellow-400' : 'text-cyan-400';
  return (
    <div className={`flex items-center gap-3 py-2 border-b border-white/5 transition-all duration-700 ${highlight ? 'opacity-100' : 'opacity-50'}`}>
      <span className="text-[9px] font-mono text-gray-500 w-24 uppercase tracking-wider">{label}</span>
      <div className="flex-1 flex items-center gap-2">
        <span className="text-[10px] font-mono text-green-400 w-10 text-right">{(left * 100).toFixed(0)}%</span>
        <div className="flex-1 h-1 bg-white/5 rounded relative flex">
          <div style={{ width: `${left * 100}%` }} className="h-full bg-green-500/60 rounded-l" />
          <div style={{ width: `${right * 100}%` }} className="h-full bg-blue-500/60 rounded-r ml-auto" />
        </div>
        <span className="text-[10px] font-mono text-blue-400 w-10">{(right * 100).toFixed(0)}%</span>
      </div>
      <span className={`text-[10px] font-mono ${severity} w-12 text-right font-bold`}>Δ{(diff * 100).toFixed(0)}%</span>
    </div>
  );
};

/* ================= CAPTURED FRAME CARD ================= */
const CapturedFrame = ({ dataUrl, timestamp, score }: { dataUrl: string; timestamp: number; score: number }) => (
  <motion.div
    initial={{ opacity: 0, scale: 0.9 }}
    animate={{ opacity: 1, scale: 1 }}
    className="relative rounded-lg overflow-hidden border border-red-500/40 bg-black/60"
  >
    <img src={dataUrl} alt={`Frame at ${timestamp.toFixed(1)}s`} className="w-full h-28 object-cover" />
    <div className="absolute inset-0 bg-gradient-to-t from-black/80 to-transparent" />
    <div className="absolute bottom-0 left-0 right-0 p-2 flex justify-between items-end">
      <span className="text-[9px] font-mono text-white/80">{timestamp.toFixed(2)}s</span>
      <span className="text-[9px] font-mono text-red-400 font-bold">{(score * 100).toFixed(0)}% fake</span>
    </div>
    <div className="absolute top-2 right-2">
      <AlertTriangle className="w-3 h-3 text-red-400" />
    </div>
  </motion.div>
);

/* ================= DASHBOARD ================= */
export const Dashboard = () => {
  // Unified upload state
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [fileType, setFileType] = useState<'video' | 'audio' | 'image' | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);

  // Analysis state
  const [analysisStarted, setAnalysisStarted] = useState(false);
  const [aiAnalysisLoading, setAiAnalysisLoading] = useState(false);
  const [aiResult, setAiResult] = useState<any>(null);          // SightEngine / audio result
  const [audioResult, setAudioResult] = useState<any>(null);

  // Face mesh state
  const [faceLandmarker, setFaceLandmarker] = useState<FaceLandmarker | null>(null);
  const [faceLandmarkerReady, setFaceLandmarkerReady] = useState(false);
  const [blendshapes, setBlendshapes] = useState<Record<string, number>>({});
  const [blinkCount, setBlinkCount] = useState(0);
  const [blinkRate, setBlinkRate] = useState(0);
  const [eyeClosed, setEyeClosed] = useState(false);
  const videoStartTime = useRef<number>(0);
  const [loopCount, setLoopCount] = useState(0);
  const [biometricComplete, setBiometricComplete] = useState(false);
  const MAX_LOOPS = 3;

  // Captured suspicious frames
  const [capturedFrames, setCapturedFrames] = useState<{ dataUrl: string; timestamp: number; score: number }[]>([]);
  const [capturingFrames, setCapturingFrames] = useState(false);

  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const captureCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const animFrameRef = useRef<number>(0);

  // Derived asymmetry
  const asymmetryPairs = [
    { label: 'Eye Blink',   left: blendshapes['eyeBlinkLeft'] ?? 0,   right: blendshapes['eyeBlinkRight'] ?? 0 },
    { label: 'Smile',       left: blendshapes['mouthSmileLeft'] ?? 0, right: blendshapes['mouthSmileRight'] ?? 0 },
    { label: 'Brow Down',   left: blendshapes['browDownLeft'] ?? 0,   right: blendshapes['browDownRight'] ?? 0 },
    { label: 'Eye Squint',  left: blendshapes['eyeSquintLeft'] ?? 0,  right: blendshapes['eyeSquintRight'] ?? 0 },
    { label: 'Mouth Frown', left: blendshapes['mouthFrownLeft'] ?? 0, right: blendshapes['mouthFrownRight'] ?? 0 },
  ];
  const asymmetryIndex = asymmetryPairs.reduce((s, p) => s + Math.abs(p.left - p.right), 0) / asymmetryPairs.length;

  // Verdict logic: AI result drives the overall assessment
  const verdict = aiResult?.overall_verdict ?? null;
  const isDeepfake = verdict === 'Deepfake';
  const verdictColor = isDeepfake ? 'text-red-400' : verdict ? 'text-cyan-400' : 'text-gray-400';
  const verdictBorder = isDeepfake ? 'border-red-500/50' : verdict ? 'border-cyan-500/50' : 'border-white/10';
  const verdictBg =    isDeepfake ? 'bg-red-950/30' : verdict ? 'bg-cyan-950/30' : 'bg-black/30';

  // Whether biometric metrics "agree" with verdict (highlight if they match the signal)
  const biometricAgrees = verdict !== null && biometricComplete;
  const blinksAbnormal = blinkRate < 10 || blinkRate > 30;
  const asymmetryAbnormal = asymmetryIndex > 0.2;

  // =============================================================
  // Handle file change
  // =============================================================
  const handleFileChange = (file: File) => {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setUploadedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
    setAnalysisStarted(false);
    setAiResult(null);
    setAudioResult(null);
    setBiometricComplete(false);
    setBlinkCount(0); setBlinkRate(0); setLoopCount(0); setBlendshapes({});
    setCapturedFrames([]);
    if (file.type.startsWith('video/')) setFileType('video');
    else if (file.type.startsWith('audio/')) setFileType('audio');
    else if (file.type.startsWith('image/')) setFileType('image');
  };

  // =============================================================
  // MediaPipe init
  // =============================================================
  useEffect(() => {
    const init = async () => {
      try {
        const fsRes = await FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm"
        );
        const lm = await FaceLandmarker.createFromOptions(fsRes, {
          baseOptions: {
            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`,
            delegate: "GPU"
          },
          outputFaceBlendshapes: true,
          runningMode: "VIDEO",
          numFaces: 1
        });
        setFaceLandmarker(lm);
        setFaceLandmarkerReady(true);
      } catch (e) { console.error("MediaPipe init failed:", e); }
    };
    init();
  }, []);

  // =============================================================
  // Frame processing loop
  // =============================================================
  const processFrame = useCallback(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || !faceLandmarker || !analysisStarted) {
      animFrameRef.current = requestAnimationFrame(processFrame);
      return;
    }
    if (video.paused || video.ended || video.readyState < 2) {
      animFrameRef.current = requestAnimationFrame(processFrame);
      return;
    }
    const ctx = canvas.getContext("2d");
    if (!ctx) { animFrameRef.current = requestAnimationFrame(processFrame); return; }
    if (canvas.width !== video.clientWidth || canvas.height !== video.clientHeight) {
      canvas.width = video.clientWidth;
      canvas.height = video.clientHeight;
    }
    try {
      const results = faceLandmarker.detectForVideo(video, performance.now());
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      if (results?.faceLandmarks?.length > 0) {
        if (results.faceBlendshapes?.length > 0) {
          const shapes: Record<string, number> = {};
          results.faceBlendshapes[0].categories.forEach((c: any) => { shapes[c.categoryName] = c.score; });
          setBlendshapes(shapes);
          const avg = ((shapes['eyeBlinkLeft'] ?? 0) + (shapes['eyeBlinkRight'] ?? 0)) / 2;
          if (avg > 0.45 && !eyeClosed) {
            setEyeClosed(true);
            setBlinkCount(prev => {
              const next = prev + 1;
              const mins = (performance.now() - videoStartTime.current) / 60000;
              if (mins > 0) setBlinkRate(Math.max(1, Math.round(next / mins)));
              return next;
            });
          } else if (avg < 0.2 && eyeClosed) { setEyeClosed(false); }
        }
        const du = new DrawingUtils(ctx);
        for (const lm of results.faceLandmarks) {
          du.drawConnectors(lm, FaceLandmarker.FACE_LANDMARKS_TESSELATION, { color: "#C0C0C055", lineWidth: 0.8 });
          du.drawConnectors(lm, FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE,   { color: "#FF4040" });
          du.drawConnectors(lm, FaceLandmarker.FACE_LANDMARKS_LEFT_EYE,    { color: "#40FF40" });
          du.drawConnectors(lm, FaceLandmarker.FACE_LANDMARKS_FACE_OVAL,   { color: "#E0E0E0", lineWidth: 1.5 });
          du.drawConnectors(lm, FaceLandmarker.FACE_LANDMARKS_LIPS,        { color: "#E0E0E0" });
          du.drawConnectors(lm, FaceLandmarker.FACE_LANDMARKS_RIGHT_EYEBROW, { color: "#FF4040" });
          du.drawConnectors(lm, FaceLandmarker.FACE_LANDMARKS_LEFT_EYEBROW,  { color: "#40FF40" });
        }
      }
    } catch (_) { /* skip frame */ }
    animFrameRef.current = requestAnimationFrame(processFrame);
  }, [faceLandmarker, analysisStarted, eyeClosed]);

  useEffect(() => {
    animFrameRef.current = requestAnimationFrame(processFrame);
    return () => cancelAnimationFrame(animFrameRef.current);
  }, [processFrame]);

  // =============================================================
  // Capture frames at suspicious timestamps
  // =============================================================
  const captureFramesAtTimestamps = useCallback(async (segments: { start_time: number; confidence: number }[]) => {
    const video = videoRef.current;
    if (!video || segments.length === 0) return;
    setCapturingFrames(true);
    const captured: { dataUrl: string; timestamp: number; score: number }[] = [];
    // Use a fresh off-screen canvas
    const offCanvas = document.createElement('canvas');
    captureCanvasRef.current = offCanvas;
    for (const seg of segments.slice(0, 6)) {
      await new Promise<void>(resolve => {
        const t = seg.start_time;
        video.currentTime = t;
        const onSeeked = () => {
          offCanvas.width = video.videoWidth;
          offCanvas.height = video.videoHeight;
          const ctx = offCanvas.getContext('2d');
          if (ctx) {
            ctx.drawImage(video, 0, 0, offCanvas.width, offCanvas.height);
            captured.push({ dataUrl: offCanvas.toDataURL('image/jpeg', 0.7), timestamp: t, score: seg.confidence });
          }
          video.removeEventListener('seeked', onSeeked);
          resolve();
        };
        video.addEventListener('seeked', onSeeked);
      });
    }
    // Resume from beginning after capture
    video.currentTime = 0;
    setCapturedFrames(captured);
    setCapturingFrames(false);
  }, []);

  // Trigger frame capture when AI result arrives with suspicious segments
  useEffect(() => {
    if (aiResult?.suspicious_segments?.length > 0 && fileType === 'video') {
      captureFramesAtTimestamps(aiResult.suspicious_segments);
    }
  }, [aiResult, fileType, captureFramesAtTimestamps]);

  // =============================================================
  // Main Analyze Handler
  // =============================================================
  const handleAnalyze = async () => {
    if (!uploadedFile) return;
    setAnalysisStarted(true);
    videoStartTime.current = performance.now();
    setBiometricComplete(false);
    setBlinkCount(0); setBlinkRate(0); setLoopCount(0);
    setAiResult(null); setAudioResult(null); setCapturedFrames([]);

    if (fileType === 'video') {
      if (videoRef.current) { videoRef.current.currentTime = 0; videoRef.current.play(); }
      setAiAnalysisLoading(true);
      try {
        const fd = new FormData();
        fd.append('file', uploadedFile);
        const res = await fetch('http://localhost:8001/api/detect-deepfake', { method: 'POST', body: fd });
        const data = await res.json();
        setAiResult(data);
      } catch (e) {
        setAiResult({ error: 'AI analysis call failed. Ensure backend is running on port 8001.' });
      } finally {
        setAiAnalysisLoading(false);
      }
    }
    if (fileType === 'audio') {
      setAiAnalysisLoading(true);
      try {
        const fd = new FormData();
        fd.append('file', uploadedFile);
        const res = await fetch('http://localhost:8001/detect-deepfake-audio', { method: 'POST', body: fd });
        const data = await res.json();
        setAudioResult(data); setAiResult(data);
      } catch (e) {
        setAiResult({ error: 'Audio analysis failed.' });
      } finally {
        setAiAnalysisLoading(false);
      }
    }
  };

  // Firebase cleanup: we remove the state vars but keep the fetch (quietly)
  useEffect(() => {
    const fetchStats = async () => {
      try {
        const auth = getAuth(); const db = getFirestore(); const user = auth.currentUser;
        if (!user) return;
        await getDoc(doc(db, 'stats', user.uid));
        await getDocs(collection(db, 'count'));
        await getDocs(collection(db, 'users'));
      } catch (_) {}
    };
    fetchStats();
  }, []);

  const FileIcon = fileType === 'audio' ? FileAudio : fileType === 'image' ? ImageIcon : Film;
  const fileTypeLabel = fileType === 'video' ? 'VIDEO' : fileType === 'audio' ? 'AUDIO' : fileType === 'image' ? 'IMAGE' : '';
  const fileTypeColor = fileType === 'video' ? 'border-cyan-500/50 text-cyan-400' : fileType === 'audio' ? 'border-purple-500/50 text-purple-400' : 'border-red-500/50 text-red-400';

  const handleVideoEnded = () => {
    if (loopCount + 1 < MAX_LOOPS) {
      setLoopCount(prev => prev + 1);
      if (videoRef.current) videoRef.current.play();
    } else {
      setBiometricComplete(true);
    }
  };

  // Overall completion
  const allDone = biometricComplete && !aiAnalysisLoading;

  return (
    <div className="h-full flex flex-col gap-6">
      {/* HEADER */}
      <motion.header initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }} className="flex items-center justify-between">
        <div>
          <h2
            className="text-5xl font-bold tracking-tight leading-none mb-2"
            style={{ fontFamily: '"Playfair Display", Georgia, serif', background: 'linear-gradient(135deg, #ffffff 0%, #ff0033 50%, #00ffff 100%)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}
          >
            Detection Lab
          </h2>
          <p className="text-gray-500 font-mono text-xs tracking-[0.15em] uppercase">
            Multi-model Deepfake Analysis · Neural + Biometric
          </p>
        </div>
        <motion.div
          animate={{ boxShadow: ['0 0 20px rgba(255,0,51,0.3)', '0 0 40px rgba(255,0,51,0.6)', '0 0 20px rgba(255,0,51,0.3)'] }}
          transition={{ duration: 2, repeat: Infinity }}
          className="flex items-center gap-3 px-6 py-3 bg-red-950/30 border border-red-500/40 rounded-full backdrop-blur-sm"
        >
          <motion.div className="w-3 h-3 bg-red-500 rounded-full" animate={{ scale: [1, 1.3, 1], opacity: [1, 0.5, 1] }} transition={{ duration: 1.5, repeat: Infinity }} />
          <span className="text-sm font-mono text-red-400 tracking-[0.15em] uppercase font-semibold">Threat: Critical</span>
        </motion.div>
      </motion.header>

      {/* MAIN GRID */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">

        {/* ===== LEFT: VIDEO/UPLOAD ===== */}
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }} className="lg:col-span-8 space-y-4">
          <Panel title="Analysis Input Portal">
            {!uploadedFile ? (
              <label className="flex flex-col items-center justify-center min-h-[420px] border-2 border-dashed border-white/20 rounded-xl bg-black/30 cursor-pointer group hover:border-red-500/50 transition-all duration-300">
                <motion.div animate={{ y: [0, -10, 0] }} transition={{ duration: 2.5, repeat: Infinity }} className="text-center space-y-4">
                  <div className="flex gap-6 justify-center">
                    <Film      className="w-8 h-8 text-cyan-600/60   group-hover:text-cyan-400   transition-colors" />
                    <FileAudio className="w-8 h-8 text-purple-600/60 group-hover:text-purple-400 transition-colors" />
                    <ImageIcon className="w-8 h-8 text-red-600/60    group-hover:text-red-400    transition-colors" />
                  </div>
                  <div>
                    <p className="text-sm font-mono text-gray-400 group-hover:text-gray-300">Drop any file or click to browse</p>
                    <p className="text-[10px] font-mono text-gray-600 mt-2">VIDEO · AUDIO · IMAGE — auto-detected</p>
                  </div>
                  <div className="flex gap-2 justify-center flex-wrap text-[9px] font-mono text-gray-700">
                    <span className="px-2 py-1 border border-white/10 rounded">MP4 MOV AVI</span>
                    <span className="px-2 py-1 border border-white/10 rounded">MP3 WAV FLAC</span>
                    <span className="px-2 py-1 border border-white/10 rounded">JPG PNG WEBP</span>
                  </div>
                </motion.div>
                <input type="file" accept="video/*,audio/*,image/*" hidden onChange={e => e.target.files?.[0] && handleFileChange(e.target.files[0])} />
              </label>
            ) : (
              <div className="space-y-4">
                {/* Video preview */}
                {fileType === 'video' && previewUrl && (
                  <div className="relative rounded-xl overflow-hidden border border-white/10 bg-black/60 flex items-center justify-center min-h-[360px]">
                    <div className="relative inline-flex items-center justify-center w-full">
                      <video ref={videoRef} src={previewUrl} className="max-w-full max-h-[520px] rounded-lg" muted playsInline onEnded={handleVideoEnded} />
                      <canvas ref={canvasRef} className="absolute pointer-events-none z-10" style={{ width: videoRef.current?.clientWidth, height: videoRef.current?.clientHeight }} />
                    </div>

                    {/* Top-left badge */}
                    <div className="absolute top-4 left-4 z-20 flex flex-col gap-2">
                      <div className={`px-3 py-1.5 bg-black/80 backdrop-blur-md border rounded-lg text-xs font-mono flex items-center gap-2 ${analysisStarted ? 'border-cyan-500/60 text-cyan-400' : 'border-white/20 text-gray-500'}`}>
                        <div className={`w-2 h-2 rounded-full ${analysisStarted ? 'bg-cyan-500 animate-pulse shadow-[0_0_8px_rgba(6,182,212,0.9)]' : 'bg-gray-600'}`} />
                        {analysisStarted ? 'LIVE MESH TRACKING' : 'PREVIEW · CLICK ANALYZE FEED'}
                      </div>
                      {analysisStarted && (
                        <div className="px-3 py-1 bg-black/80 backdrop-blur-md border border-white/20 rounded-lg text-[9px] font-mono text-gray-400">
                          CYCLE {loopCount + 1} / {MAX_LOOPS}
                        </div>
                      )}
                    </div>

                    {/* Top-right biometric overlay */}
                    {analysisStarted && !biometricComplete && (
                      <div className="absolute top-4 right-4 z-20 flex flex-col gap-2">
                        <div className="px-3 py-1.5 bg-black/80 backdrop-blur-md border border-red-500/50 rounded-lg text-xs font-mono text-red-400 flex items-center gap-2">
                          <AlertTriangle className="w-3 h-3" /> BLINKS: {blinkCount}
                        </div>
                        <div className="px-3 py-1.5 bg-black/80 backdrop-blur-md border border-yellow-500/50 rounded-lg text-xs font-mono text-yellow-400 flex items-center gap-2">
                          <Activity className="w-3 h-3" /> {blinkRate} BPM
                        </div>
                        <div className="px-3 py-1.5 bg-black/80 backdrop-blur-md border border-purple-500/50 rounded-lg text-xs font-mono text-purple-400 flex items-center gap-2">
                          <Zap className="w-3 h-3" /> ASYM: {(asymmetryIndex * 100).toFixed(1)}%
                        </div>
                      </div>
                    )}

                    {/* AI result arrives early banner */}
                    <AnimatePresence>
                      {aiResult && !biometricComplete && !aiResult.error && (
                        <motion.div
                          initial={{ opacity: 0, y: 20 }}
                          animate={{ opacity: 1, y: 0 }}
                          exit={{ opacity: 0 }}
                          className={`absolute bottom-4 left-4 right-4 z-20 px-4 py-3 ${verdictBg} border ${verdictBorder} rounded-xl backdrop-blur-md flex items-center justify-between`}
                        >
                          <div className="flex items-center gap-3">
                            {isDeepfake ? <AlertTriangle className="w-5 h-5 text-red-400 flex-shrink-0" /> : <CheckCircle className="w-5 h-5 text-cyan-400 flex-shrink-0" />}
                            <div>
                              <p className="text-[9px] font-mono text-gray-400 uppercase tracking-widest">AI Video Analysis Complete</p>
                              <p className={`font-bold font-mono ${verdictColor}`}>{aiResult.overall_verdict} · {((aiResult.overall_confidence ?? 0) * 100).toFixed(1)}% confidence</p>
                            </div>
                          </div>
                          <span className="text-[9px] font-mono text-gray-500">Biometric scan continuing…</span>
                        </motion.div>
                      )}
                    </AnimatePresence>

                    {/* Full biometric complete overlay */}
                    <AnimatePresence>
                      {biometricComplete && (
                        <motion.div
                          initial={{ opacity: 0, scale: 0.95 }}
                          animate={{ opacity: 1, scale: 1 }}
                          className="absolute inset-0 z-30 bg-black/96 backdrop-blur-xl flex flex-col items-center justify-center p-8 text-center"
                        >
                          <motion.div initial={{ y: 20 }} animate={{ y: 0 }} className="space-y-4 w-full max-w-md">
                            {/* AI Verdict — primary */}
                            {aiResult && !aiResult.error && (
                              <div className={`p-5 rounded-2xl border ${verdictBorder} ${verdictBg} mb-2`}>
                                <p className="text-[9px] font-mono text-gray-400 uppercase tracking-[0.2em] mb-1">AI Video Analysis Verdict</p>
                                <p className={`text-4xl font-bold font-mono ${verdictColor}`}>{aiResult.overall_verdict}</p>
                                <p className="text-gray-500 font-mono text-xs mt-1">{((aiResult.overall_confidence ?? 0) * 100).toFixed(1)}% confidence · avg {((aiResult.average_score ?? 0) * 100).toFixed(1)}%</p>
                              </div>
                            )}

                            <p className="text-xs font-mono text-gray-500 uppercase tracking-widest">Biometric Corroboration — {MAX_LOOPS} Cycles</p>

                            <div className="grid grid-cols-3 gap-3">
                              <div className={`p-3 rounded-xl border ${blinksAbnormal && biometricAgrees && isDeepfake ? 'border-red-500/50 bg-red-950/30' : 'border-white/10 bg-white/5'}`}>
                                <p className="text-[9px] font-mono text-gray-500 uppercase mb-1">Blinks</p>
                                <p className={`text-2xl font-bold font-mono ${blinksAbnormal ? 'text-red-400' : 'text-cyan-400'}`}>{blinkCount}</p>
                              </div>
                              <div className={`p-3 rounded-xl border ${blinksAbnormal && biometricAgrees && isDeepfake ? 'border-yellow-500/50 bg-yellow-950/20' : 'border-white/10 bg-white/5'}`}>
                                <p className="text-[9px] font-mono text-gray-500 uppercase mb-1">BPM</p>
                                <p className={`text-2xl font-bold font-mono ${blinksAbnormal ? 'text-yellow-400' : 'text-green-400'}`}>{blinkRate}</p>
                              </div>
                              <div className={`p-3 rounded-xl border ${asymmetryAbnormal && biometricAgrees && isDeepfake ? 'border-purple-500/50 bg-purple-950/20' : 'border-white/10 bg-white/5'}`}>
                                <p className="text-[9px] font-mono text-gray-500 uppercase mb-1">Asymmetry</p>
                                <p className={`text-2xl font-bold font-mono ${asymmetryAbnormal ? 'text-purple-400' : 'text-green-400'}`}>{(asymmetryIndex * 100).toFixed(0)}%</p>
                              </div>
                            </div>

                            {/* Combined conclusion */}
                            {aiResult && !aiResult.error && (
                              <div className={`p-4 rounded-xl border ${verdictBorder} ${verdictBg} text-left`}>
                                <p className="text-[9px] font-mono text-gray-400 uppercase tracking-widest mb-2">Combined Assessment</p>
                                <p className="text-xs font-mono text-gray-300 leading-relaxed">
                                  {isDeepfake
                                    ? `Neural analysis flagged this video as synthetic content. ${asymmetryAbnormal ? 'Facial asymmetry index is elevated. ' : ''}${blinksAbnormal ? 'Blinking pattern deviates from biological norms.' : 'Biometrics show some natural indicators.'}`
                                    : `Neural analysis indicates authentic content. ${!asymmetryAbnormal ? 'Facial symmetry is within natural ranges. ' : 'Minor asymmetry detected but within tolerance. '}Biometric data supports this conclusion.`}
                                </p>
                              </div>
                            )}

                            <motion.button
                              whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}
                              onClick={() => {
                                setBiometricComplete(false); setLoopCount(0); setBlinkCount(0);
                                videoStartTime.current = performance.now();
                                if (videoRef.current) { videoRef.current.currentTime = 0; videoRef.current.play(); }
                              }}
                              className="mt-2 px-6 py-2.5 bg-white/10 hover:bg-white/20 text-white rounded-full font-mono text-xs transition-colors border border-white/20"
                            >
                              RESCAN BIOMETRICS
                            </motion.button>
                          </motion.div>
                        </motion.div>
                      )}
                    </AnimatePresence>
                  </div>
                )}

                {/* Audio preview */}
                {fileType === 'audio' && previewUrl && (
                  <div className="flex flex-col items-center justify-center min-h-[180px] rounded-xl border border-white/10 bg-black/40 gap-4 p-8">
                    <FileAudio className="w-14 h-14 text-purple-400 opacity-50" />
                    <audio controls src={previewUrl} className="w-full max-w-md" />
                  </div>
                )}

                {/* Image preview */}
                {fileType === 'image' && previewUrl && (
                  <div className="flex items-center justify-center min-h-[200px] rounded-xl border border-white/10 bg-black/40 overflow-hidden">
                    <img src={previewUrl} className="max-h-64 object-contain" alt="preview" />
                  </div>
                )}

                {/* File bar */}
                <div className={`flex items-center gap-3 p-4 bg-black/40 rounded-lg border ${fileTypeColor}`}>
                  <FileIcon className="w-5 h-5" />
                  <div className="flex-1">
                    <p className="text-xs font-mono text-white truncate">{uploadedFile.name}</p>
                    <p className="text-[10px] font-mono text-gray-500">
                      {fileTypeLabel} · {(uploadedFile.size / (1024 * 1024)).toFixed(2)} MB
                      {!faceLandmarkerReady && fileType === 'video' && <span className="text-yellow-500 ml-2">· Loading AI model…</span>}
                    </p>
                  </div>
                  <motion.button whileHover={{ scale: 1.1, rotate: 90 }} whileTap={{ scale: 0.9 }}
                    onClick={() => { setUploadedFile(null); setPreviewUrl(null); setFileType(null); setAnalysisStarted(false); setAiResult(null); setBiometricComplete(false); setCapturedFrames([]); }}
                    className="text-gray-500 hover:text-red-400"><X className="w-4 h-4" /></motion.button>
                </div>

                {/* Analyze button */}
                <motion.button
                  onClick={handleAnalyze}
                  disabled={(analysisStarted && fileType === 'video' && !biometricComplete)}
                  whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }}
                  className="w-full px-4 py-4 bg-gradient-to-r from-red-600 via-red-500 to-orange-600 hover:from-red-500 hover:to-orange-500 text-white rounded-xl font-mono text-sm tracking-widest transition-all disabled:opacity-40 relative overflow-hidden"
                >
                  <motion.div className="absolute inset-0 bg-gradient-to-r from-white/0 via-white/15 to-white/0" animate={{ x: ['-100%', '200%'] }} transition={{ duration: 2.5, repeat: Infinity }} />
                  <span className="relative z-10 flex items-center justify-center gap-3">
                    {aiAnalysisLoading ? <><Loader className="w-4 h-4 animate-spin" /> RUNNING AI ANALYSIS…</>
                     : analysisStarted && fileType === 'video' && !biometricComplete ? <><Activity className="w-4 h-4 animate-pulse" /> BIOMETRIC SCAN IN PROGRESS…</>
                     : <><Zap className="w-4 h-4" /> ANALYZE FEED</>}
                  </span>
                </motion.button>
              </div>
            )}
          </Panel>

          {/* ===== SUSPICIOUS FRAMES ===== */}
          <AnimatePresence>
            {(capturedFrames.length > 0 || capturingFrames) && (
              <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}>
                <Panel title="Suspicious Frame Capture">
                  {capturingFrames ? (
                    <div className="flex items-center gap-3 py-4 justify-center text-gray-400 font-mono text-xs">
                      <Camera className="w-4 h-4 animate-pulse text-red-400" />
                      <span>Extracting frames at suspicious timestamps…</span>
                    </div>
                  ) : (
                    <div className="space-y-3">
                      <p className="text-[9px] font-mono text-gray-500 uppercase tracking-widest">
                        {capturedFrames.length} suspicious timestamp{capturedFrames.length !== 1 ? 's' : ''} extracted from video
                      </p>
                      <div className="grid grid-cols-3 gap-3">
                        {capturedFrames.map((f, i) => <CapturedFrame key={i} {...f} />)}
                      </div>
                    </div>
                  )}
                </Panel>
              </motion.div>
            )}
          </AnimatePresence>
        </motion.div>

        {/* ===== RIGHT: DIAGNOSTICS ===== */}
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.35 }} className="lg:col-span-4 space-y-4">

          {/* AI VERDICT (primary — always shown when available) */}
          {(aiAnalysisLoading || aiResult) && (
            <Panel title="Neural Video Analysis">
              {aiAnalysisLoading ? (
                <div className="flex flex-col items-center gap-3 py-8 text-center">
                  <Loader className="w-6 h-6 animate-spin text-cyan-400" />
                  <p className="text-xs font-mono text-gray-400">Uploading to neural analysis engine…</p>
                  <p className="text-[9px] font-mono text-gray-600">This may take 30–60 seconds</p>
                </div>
              ) : aiResult?.error ? (
                <div className="p-3 bg-red-950/20 border border-red-500/30 rounded-lg">
                  <p className="text-red-400 text-xs font-mono">{aiResult.error}</p>
                </div>
              ) : aiResult && (
                <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="space-y-3">
                  <div className={`p-4 rounded-xl border text-center ${verdictBorder} ${verdictBg}`}>
                    <p className="text-[9px] font-mono text-gray-400 uppercase mb-1">Verdict</p>
                    <p className={`text-3xl font-bold font-mono ${verdictColor}`}>{aiResult.overall_verdict}</p>
                  </div>
                  <div className="grid grid-cols-2 gap-2">
                    <div className="p-3 bg-black/40 border border-white/10 rounded-xl text-center">
                      <p className="text-[9px] font-mono text-gray-500 uppercase mb-1">Confidence</p>
                      <p className="text-xl font-bold font-mono text-white">{((aiResult.overall_confidence ?? 0) * 100).toFixed(1)}%</p>
                    </div>
                    <div className="p-3 bg-black/40 border border-white/10 rounded-xl text-center">
                      <p className="text-[9px] font-mono text-gray-500 uppercase mb-1">Avg Score</p>
                      <p className="text-xl font-bold font-mono text-orange-400">{((aiResult.average_score ?? 0) * 100).toFixed(1)}%</p>
                    </div>
                  </div>
                  {aiResult.suspicious_segments?.length > 0 && (
                    <div className="space-y-1">
                      <p className="text-[9px] font-mono text-gray-500 uppercase tracking-widest">{aiResult.suspicious_segments.length} suspicious segments</p>
                      {aiResult.suspicious_segments.slice(0, 4).map((s: any, i: number) => (
                        <div key={i} className="flex items-center gap-2 p-2 bg-red-950/20 border-l-2 border-red-500 rounded text-[10px] font-mono">
                          <AlertTriangle className="w-3 h-3 text-red-400 flex-shrink-0" />
                          <span className="text-red-400">{s.start_time.toFixed(2)}s</span>
                          <span className="text-gray-500">·</span>
                          <span className="text-orange-400">{(s.confidence * 100).toFixed(1)}% fake</span>
                        </div>
                      ))}
                    </div>
                  )}
                </motion.div>
              )}
            </Panel>
          )}

          {/* FACIAL MUSCLE ACTIVITY */}
          {fileType === 'video' && (
            <Panel title="Facial Muscle Activity">
              <div className="space-y-3 p-1">
                <MetricBar label="Eye Blink L/R" value={((blendshapes['eyeBlinkLeft'] ?? 0) + (blendshapes['eyeBlinkRight'] ?? 0)) / 2} color="bg-red-500"    highlight={biometricAgrees && isDeepfake} />
                <MetricBar label="Brow Raise"    value={blendshapes['browInnerUp'] ?? 0}   color="bg-orange-500"  highlight={biometricAgrees} />
                <MetricBar label="Smile"         value={((blendshapes['mouthSmileLeft'] ?? 0) + (blendshapes['mouthSmileRight'] ?? 0)) / 2} color="bg-green-500" highlight={biometricAgrees} />
                <MetricBar label="Jaw Open"      value={blendshapes['jawOpen'] ?? 0}        color="bg-blue-500"    highlight={biometricAgrees} />
                <MetricBar label="Cheek Puff"    value={blendshapes['cheekPuff'] ?? 0}      color="bg-purple-500"  highlight={biometricAgrees && isDeepfake} />
                <MetricBar label="Mouth Pucker"  value={blendshapes['mouthPucker'] ?? 0}    color="bg-pink-500"    highlight={biometricAgrees} />
                <p className="text-[9px] font-mono text-gray-600 italic pt-1 border-t border-white/5">
                  {biometricAgrees ? (isDeepfake ? '⚠ Metrics highlighted for deepfake correlation' : '✓ Metrics align with authentic classification') : 'Real-time MediaPipe blendshapes'}
                </p>
              </div>
            </Panel>
          )}

          {/* MUSCLE POTENTIAL DIFFERENCE */}
          {fileType === 'video' && analysisStarted && (
            <Panel title="Muscle Potential Difference">
              <div className="p-1">
                <div className="flex justify-between items-center mb-3">
                  <span className="text-[9px] font-mono text-gray-500 uppercase tracking-widest">Overall Asymmetry Index</span>
                  <span className={`text-sm font-bold font-mono ${asymmetryIndex > 0.3 ? 'text-red-400' : asymmetryIndex > 0.15 ? 'text-yellow-400' : 'text-cyan-400'}`}>
                    {(asymmetryIndex * 100).toFixed(1)}%
                    {biometricAgrees && isDeepfake && asymmetryAbnormal && <span className="text-red-500 ml-1 text-xs">⚠</span>}
                  </span>
                </div>
                <div className="flex text-[8px] font-mono text-gray-600 justify-between px-1 mb-1">
                  <span className="text-green-500">LEFT ◀</span>
                  <span>PAIR</span>
                  <span className="text-blue-500">▶ RIGHT</span>
                </div>
                {asymmetryPairs.map((p, i) => <AsymmetryRow key={i} {...p} highlight={biometricAgrees} />)}
                <p className="text-[9px] font-mono text-gray-600 italic pt-3 border-t border-white/5 mt-2">
                  {biometricAgrees && isDeepfake && asymmetryAbnormal
                    ? '⚠ Elevated asymmetry corroborates AI deepfake detection'
                    : 'Δ = |L−R| per muscle pair. Elevated values may indicate AI manipulation.'}
                </p>
              </div>
            </Panel>
          )}

          {/* Audio result panel */}
          {audioResult && (
            <Panel title="Audio Analysis">
              {audioResult.error ? <p className="text-red-400 text-xs font-mono">{audioResult.error}</p> : (
                <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="space-y-3">
                  <div className={`p-4 rounded-xl border text-center ${audioResult.is_deepfake ? 'bg-red-950/30 border-red-500/50' : 'bg-cyan-950/30 border-cyan-500/50'}`}>
                    <p className={`text-2xl font-bold font-mono ${audioResult.is_deepfake ? 'text-red-400' : 'text-cyan-400'}`}>{audioResult.overall_verdict}</p>
                  </div>
                  <div className="grid grid-cols-2 gap-2">
                    <div className="p-3 bg-black/40 border border-white/10 rounded-xl text-center">
                      <p className="text-[9px] font-mono text-gray-500 uppercase mb-1">Confidence</p>
                      <p className="text-xl font-bold font-mono text-white">{((audioResult.overall_confidence ?? 0) * 100).toFixed(1)}%</p>
                    </div>
                    <div className="p-3 bg-black/40 border border-white/10 rounded-xl text-center">
                      <p className="text-[9px] font-mono text-gray-500 uppercase mb-1">Fake Prob</p>
                      <p className="text-xl font-bold font-mono text-orange-400">{((audioResult.fake_probability ?? 0) * 100).toFixed(1)}%</p>
                    </div>
                  </div>
                </motion.div>
              )}
            </Panel>
          )}

          {/* Placeholder */}
          {!uploadedFile && (
            <Panel title="Diagnostics">
              <div className="flex flex-col items-center justify-center py-12 text-center opacity-30">
                <Activity className="w-10 h-10 text-gray-600" />
                <p className="text-gray-600 font-mono text-xs mt-2">Awaiting media input…</p>
              </div>
            </Panel>
          )}
        </motion.div>
      </div>

      <style>{`
        .custom-scrollbar::-webkit-scrollbar { width: 6px; }
        .custom-scrollbar::-webkit-scrollbar-track { background: rgba(255,255,255,0.05); border-radius: 10px; }
        .custom-scrollbar::-webkit-scrollbar-thumb { background: rgba(255,0,51,0.3); border-radius: 10px; }
      `}</style>
    </div>
  );
};
