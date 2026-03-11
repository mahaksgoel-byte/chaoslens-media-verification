import { motion } from "framer-motion";
import {
  ScanFace,
  FileSearch,
  ArrowRight,
  EyeOff,
  Siren,
  Shield,
  Zap,
  X
} from "lucide-react";
import { useNavigate } from "react-router-dom";
import { useState } from "react";

/* ================= ANIMATIONS ================= */
const fadeUp = {
  hidden: { opacity: 0, y: 40 },
  visible: (i = 0) => ({
    opacity: 1,
    y: 0,
    transition: {
      duration: 0.8,
      delay: i * 0.15,
      ease: [0.22, 1, 0.36, 1],
    },
  }),
};

const scaleIn = {
  hidden: { opacity: 0, scale: 0.8 },
  visible: {
    opacity: 1,
    scale: 1,
    transition: {
      duration: 0.6,
      ease: [0.22, 1, 0.36, 1],
    },
  },
};

/* ================= FLOATING PARTICLES ================= */
const FloatingParticles = () => {
  return (
    <div className="absolute inset-0 overflow-hidden pointer-events-none">
      {[...Array(30)].map((_, i) => (
        <motion.div
          key={i}
          className="absolute w-1 h-1 bg-red-500 rounded-full"
          style={{
            left: `${Math.random() * 100}%`,
            top: `${Math.random() * 100}%`,
          }}
          animate={{
            y: [0, -30, 0],
            opacity: [0.2, 0.8, 0.2],
            scale: [1, 1.5, 1],
          }}
          transition={{
            duration: 3 + Math.random() * 2,
            repeat: Infinity,
            delay: Math.random() * 2,
          }}
        />
      ))}
    </div>
  );
};

export const HomePage = () => {
  const [showAgenda, setShowAgenda] = useState(false);
  const navigate = useNavigate();

  return (
    <div className="min-h-screen bg-black text-white overflow-hidden relative">
      <FloatingParticles />

      {/* ================= BACKGROUND IMAGE ================= */}
      <div 
        className="pointer-events-none fixed inset-0 bg-cover bg-center bg-no-repeat opacity-60"
        style={{
          backgroundImage: 'url(/home.png)',
        }}
      />
      <div className="pointer-events-none fixed inset-0 bg-black/60" />

      {/* ================= HERO SECTION ================= */}
      <section className="relative z-10 border-b border-white/5">
        <div className="max-w-7xl mx-auto px-6 py-32">
          <div className="grid lg:grid-cols-2 gap-20 items-center">
            {/* LEFT CONTENT */}
            <div className="space-y-8">
              <motion.div
                variants={fadeUp}
                initial="hidden"
                animate="visible"
                custom={0}
                className="inline-flex items-center gap-3 px-5 py-2.5 border border-red-500/30 bg-red-950/20 backdrop-blur-sm rounded-full"
              >
                <motion.div
                  animate={{ rotate: [0, 360] }}
                  transition={{ duration: 3, repeat: Infinity, ease: "linear" }}
                >
                  <Siren className="w-4 h-4 text-red-400" />
                </motion.div>
                <span className="text-red-400 text-[10px] font-mono tracking-[0.25em] uppercase font-semibold">
                  Reality Integrity Warning
                </span>
              </motion.div>

              <motion.h1
                variants={fadeUp}
                initial="hidden"
                animate="visible"
                custom={1}
                className="text-6xl lg:text-7xl font-bold leading-[1.1]"
                style={{
                  fontFamily: '"Playfair Display", Georgia, serif',
                }}
              >
                The Age of AI Has
                <br />
                <span
                  className="block mt-2"
                  style={{
                    background: 'linear-gradient(135deg, #ff0033 0%, #ff0066 100%)',
                    WebkitBackgroundClip: 'text',
                    WebkitTextFillColor: 'transparent',
                  }}
                >
                  Broken Trust in Reality
                </span>
              </motion.h1>

              <motion.p
                variants={fadeUp}
                initial="hidden"
                animate="visible"
                custom={2}
                className="text-gray-400 text-lg leading-relaxed max-w-xl"
                style={{ fontFamily: '"Crimson Text", Georgia, serif' }}
              >
                Deepfakes, synthetic voices, and AI-generated misinformation now
                spread faster than truth. This platform exists to detect,
                analyze, and expose what is real — before damage is done.
              </motion.p>

              <motion.div
                variants={fadeUp}
                initial="hidden"
                animate="visible"
                custom={3}
                className="flex flex-wrap gap-4"
              >
                <motion.button
                  onClick={() => navigate("/home/deepfake")}
                  whileHover={{ scale: 1.05, boxShadow: '0 0 30px rgba(255,0,51,0.4)' }}
                  whileTap={{ scale: 0.95 }}
                  className="group relative px-8 py-4 bg-gradient-to-r from-red-600 to-red-700 text-white font-mono text-xs tracking-[0.2em] uppercase font-semibold rounded-xl overflow-hidden"
                >
                  <motion.div
                    className="absolute inset-0 bg-gradient-to-r from-white/0 via-white/20 to-white/0"
                    animate={{ x: ['-100%', '200%'] }}
                    transition={{ duration: 2, repeat: Infinity }}
                  />
                  <span className="relative z-10 flex items-center gap-2">
                    Enter Detection Lab
                    <motion.div
                      animate={{ x: [0, 5, 0] }}
                      transition={{ duration: 1.5, repeat: Infinity }}
                    >
                      <ArrowRight className="w-4 h-4" />
                    </motion.div>
                  </span>
                </motion.button>

                <motion.button
                  onClick={() => setShowAgenda(true)}
                  whileHover={{ scale: 1.05, borderColor: 'rgba(255,255,255,0.4)' }}
                  whileTap={{ scale: 0.95 }}
                  className="px-8 py-4 border border-white/20 text-gray-300 font-mono text-xs tracking-[0.2em] uppercase hover:bg-white/5 rounded-xl transition-all"
                >
                  Learn Why This Matters
                </motion.button>
              </motion.div>
            </div>

            {/* RIGHT - STATS */}
            <motion.div
              variants={scaleIn}
              initial="hidden"
              animate="visible"
              className="relative"
            >
              <div className="relative border border-white/10 bg-black/40 backdrop-blur-sm rounded-2xl p-8 overflow-hidden">
                {/* Glow effect */}
                <div className="absolute inset-0 bg-gradient-to-br from-red-500/5 to-transparent" />

                <p className="text-[9px] font-mono text-gray-600 tracking-[0.25em] uppercase mb-6 relative z-10">
                  Global Threat Snapshot
                </p>

                <div className="grid grid-cols-2 gap-4 relative z-10">
                  {[
                    { label: "Daily Deepfakes", value: "↑ 900%", color: "text-red-500" },
                    { label: "Misinfo Campaigns", value: "Active", color: "text-cyan-400" },
                    { label: "Voice Clones", value: "Untraceable", color: "text-purple-400" },
                    { label: "Public Trust", value: "Declining", color: "text-gray-500" },
                  ].map((stat, i) => (
                    <motion.div
                      key={i}
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: 0.6 + i * 0.1 }}
                      whileHover={{ y: -4 }}
                      className="border border-white/10 bg-black/40 backdrop-blur-sm rounded-lg p-5 group cursor-default"
                    >
                      <p className="text-[9px] font-mono text-gray-600 tracking-wider uppercase mb-2">
                        {stat.label}
                      </p>
                      <p className={`text-2xl font-bold ${stat.color} font-mono`}>
                        {stat.value}
                      </p>
                    </motion.div>
                  ))}
                </div>

                {/* Animated corner accents */}
                {[
                  { top: 0, left: 0 },
                  { top: 0, right: 0 },
                  { bottom: 0, left: 0 },
                  { bottom: 0, right: 0 },
                ].map((pos, i) => (
                  <motion.div
                    key={i}
                    className="absolute w-8 h-8 border-red-500/30"
                    style={{
                      ...pos,
                      borderTopWidth: pos.top === 0 ? '2px' : 0,
                      borderBottomWidth: pos.bottom === 0 ? '2px' : 0,
                      borderLeftWidth: pos.left === 0 ? '2px' : 0,
                      borderRightWidth: pos.right === 0 ? '2px' : 0,
                    }}
                    animate={{ opacity: [0.3, 0.8, 0.3] }}
                    transition={{ duration: 2, repeat: Infinity, delay: i * 0.5 }}
                  />
                ))}
              </div>
            </motion.div>
          </div>
        </div>
      </section>

      {/* ================= CORE CAPABILITIES ================= */}
      <section className="relative z-10 py-32">
        <div className="max-w-7xl mx-auto px-6">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.8 }}
            className="text-center mb-20"
          >
            <h2
              className="text-5xl font-bold mb-6"
              style={{ fontFamily: '"Playfair Display", Georgia, serif' }}
            >
              Two Fronts. One Mission.
            </h2>
            <p className="text-gray-500 font-mono text-sm tracking-[0.15em] uppercase max-w-2xl mx-auto">
              Reality must be defended at both the media level and the factual level.
            </p>
          </motion.div>

          <div className="grid md:grid-cols-2 gap-8">
            {/* DEEPFAKE */}
            <motion.div
              onClick={() => navigate("/home/deepfake")}
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: 0.2 }}
              whileHover={{ y: -8, boxShadow: '0 20px 60px rgba(255,0,51,0.2)' }}
              className="group cursor-pointer border border-white/10 bg-black/40 backdrop-blur-sm rounded-2xl p-10 hover:border-red-500/40 transition-all relative overflow-hidden"
            >
              <motion.div
                className="absolute inset-0 bg-gradient-to-br from-red-500/5 to-transparent opacity-0 group-hover:opacity-100"
                transition={{ duration: 0.3 }}
              />

              <motion.div
                animate={{ rotate: [0, 5, -5, 0] }}
                transition={{ duration: 3, repeat: Infinity }}
                className="w-16 h-16 rounded-2xl bg-red-950/30 flex items-center justify-center mb-6 relative z-10"
              >
                <ScanFace className="w-8 h-8 text-red-500" />
              </motion.div>

              <h3
                className="text-2xl font-semibold mb-4 relative z-10"
                style={{ fontFamily: '"Playfair Display", Georgia, serif' }}
              >
                Deepfake Detection
              </h3>

              <p className="text-gray-400 leading-relaxed mb-4 relative z-10" style={{ fontFamily: '"Crimson Text", Georgia, serif' }}>
                Analyze video, audio, and visual media for synthetic artifacts,
                facial inconsistencies, voice cloning, and manipulation traces
                invisible to the human eye.
              </p>

              <p className="text-xs text-gray-600 font-mono relative z-10">
                Used against fake scandals, impersonation, fraud, and political manipulation.
              </p>

              <div className="absolute bottom-6 right-6 opacity-0 group-hover:opacity-100 transition-opacity">
                <ArrowRight className="w-6 h-6 text-red-500" />
              </div>
            </motion.div>

            {/* MISINFO */}
            <motion.div
              onClick={() => navigate("/home/misinfo")}
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: 0.3 }}
              whileHover={{ y: -8, boxShadow: '0 20px 60px rgba(0,255,255,0.2)' }}
              className="group cursor-pointer border border-white/10 bg-black/40 backdrop-blur-sm rounded-2xl p-10 hover:border-cyan-500/40 transition-all relative overflow-hidden"
            >
              <motion.div
                className="absolute inset-0 bg-gradient-to-br from-cyan-500/5 to-transparent opacity-0 group-hover:opacity-100"
                transition={{ duration: 0.3 }}
              />

              <motion.div
                animate={{ rotate: [0, -5, 5, 0] }}
                transition={{ duration: 3, repeat: Infinity, delay: 0.5 }}
                className="w-16 h-16 rounded-2xl bg-cyan-950/30 flex items-center justify-center mb-6 relative z-10"
              >
                <FileSearch className="w-8 h-8 text-cyan-400" />
              </motion.div>

              <h3
                className="text-2xl font-semibold mb-4 relative z-10"
                style={{ fontFamily: '"Playfair Display", Georgia, serif' }}
              >
                Misinformation & Fact Analysis
              </h3>

              <p className="text-gray-400 leading-relaxed mb-4 relative z-10" style={{ fontFamily: '"Crimson Text", Georgia, serif' }}>
                Break down claims, detect narrative manipulation, verify sources,
                and compare content against known factual databases and patterns
                of coordinated misinformation.
              </p>

              <p className="text-xs text-gray-600 font-mono relative z-10">
                Designed for journalists, researchers, investigators, and citizens.
              </p>

              <div className="absolute bottom-6 right-6 opacity-0 group-hover:opacity-100 transition-opacity">
                <ArrowRight className="w-6 h-6 text-cyan-400" />
              </div>
            </motion.div>
          </div>
        </div>
      </section>

      {/* ================= WHY THIS MATTERS ================= */}
      <section className="relative z-10 py-32 border-t border-white/5">
        <div className="max-w-4xl mx-auto px-6 text-center">
          <motion.div
            initial={{ opacity: 0, scale: 0.8 }}
            whileInView={{ opacity: 1, scale: 1 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6 }}
            className="mb-8"
          >
            <div className="inline-flex items-center justify-center w-20 h-20 rounded-full bg-red-950/30 border border-red-500/30 mb-6">
              <EyeOff className="w-10 h-10 text-red-500" />
            </div>
          </motion.div>

          <motion.h2
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-4xl font-bold mb-8"
            style={{ fontFamily: '"Playfair Display", Georgia, serif' }}
          >
            When You Can't Trust What You See or Hear
          </motion.h2>

          <motion.p
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ delay: 0.2 }}
            className="text-gray-400 text-lg leading-relaxed mb-6"
            style={{ fontFamily: '"Crimson Text", Georgia, serif' }}
          >
            AI has removed the cost of lying at scale. Anyone can now fabricate
            evidence, speeches, confessions, or events. Without verification
            systems, truth collapses — and with it, democracy, justice, and trust.
          </motion.p>

          <motion.p
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ delay: 0.3 }}
            className="text-gray-600 font-mono text-sm tracking-wide"
          >
            This platform exists to restore friction, verification, and accountability.
          </motion.p>
        </div>
      </section>

      {/* ================= FINAL CTA ================= */}
      <section className="relative z-10 py-20">
        <div className="max-w-6xl mx-auto px-6">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="relative border border-white/10 bg-black/40 backdrop-blur-sm rounded-2xl p-12 overflow-hidden"
          >
            <div className="absolute inset-0 bg-gradient-to-r from-red-500/5 to-cyan-500/5" />

            <div className="relative z-10 flex flex-col md:flex-row items-center justify-between gap-8">
              <div>
                <p className="text-[10px] font-mono text-gray-600 tracking-[0.25em] uppercase mb-3">
                  System Ready
                </p>
                <h3
                  className="text-3xl font-semibold mb-3"
                  style={{ fontFamily: '"Playfair Display", Georgia, serif' }}
                >
                  Enter the Detection Lab
                </h3>
                <p className="text-gray-500 font-mono text-sm">
                  Analyze media. Verify claims. Defend reality.
                </p>
              </div>

              <motion.button
                onClick={() => navigate("/home/deepfake")}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                className="group relative px-8 py-4 bg-gradient-to-r from-red-600 to-red-700 text-white font-mono text-xs tracking-[0.2em] uppercase font-semibold rounded-xl overflow-hidden"
              >
                <motion.div
                  className="absolute inset-0 bg-gradient-to-r from-white/0 via-white/20 to-white/0"
                  animate={{ x: ['-100%', '200%'] }}
                  transition={{ duration: 2, repeat: Infinity }}
                />
                <span className="relative z-10 flex items-center gap-2">
                  Deploy System
                  <ArrowRight className="w-4 h-4" />
                </span>
              </motion.button>
            </div>
          </motion.div>
        </div>
      </section>

      {/* ================= FOOTER ================= */}
      <footer className="border-t border-white/5 py-8 text-center">
        <p className="text-[9px] font-mono text-gray-700 tracking-[0.2em] uppercase">
          Reality Defense System © {new Date().getFullYear()}
        </p>
      </footer>

      {/* ================= AGENDA MODAL ================= */}
      {showAgenda && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-md"
          onClick={() => setShowAgenda(false)}
        >
          <motion.div
            initial={{ scale: 0.9, y: 20 }}
            animate={{ scale: 1, y: 0 }}
            exit={{ scale: 0.9, y: 20 }}
            onClick={(e) => e.stopPropagation()}
            className="relative max-w-2xl w-full mx-4 border border-white/10 bg-gradient-to-br from-black via-red-950/10 to-black rounded-2xl overflow-hidden"
            style={{
              boxShadow: '0 0 80px rgba(255,0,51,0.3)',
            }}
          >
            {/* Header */}
            <div className="px-8 py-6 border-b border-white/10 flex items-center justify-between">
              <div>
                <p className="text-[9px] font-mono text-gray-600 tracking-[0.25em] uppercase mb-2">
                  Platform Agenda
                </p>
                <h3
                  className="text-2xl font-bold"
                  style={{ fontFamily: '"Playfair Display", Georgia, serif' }}
                >
                  Why This System Exists
                </h3>
              </div>
              <motion.button
                whileHover={{ scale: 1.1, rotate: 90 }}
                whileTap={{ scale: 0.9 }}
                onClick={() => setShowAgenda(false)}
                className="text-gray-500 hover:text-white transition-colors"
              >
                <X className="w-5 h-5" />
              </motion.button>
            </div>

            {/* Content */}
            <div className="px-8 py-8 space-y-6">
              <p className="text-gray-400 leading-relaxed" style={{ fontFamily: '"Crimson Text", Georgia, serif', fontSize: '1.1rem' }}>
                Artificial intelligence has eliminated the cost of deception.
                Fake videos, cloned voices, and manufactured evidence can now
                destabilize reputations, elections, and justice systems in minutes.
              </p>

              <p className="text-gray-400 leading-relaxed" style={{ fontFamily: '"Crimson Text", Georgia, serif', fontSize: '1.1rem' }}>
                This platform exists to analyze, verify, and expose manipulated media
                and misinformation before it causes irreversible harm.
              </p>

              <div className="pt-4 border-t border-white/10">
                <p className="text-xs font-mono text-gray-600 tracking-[0.15em] uppercase">
                  Detect • Verify • Defend Reality
                </p>
              </div>
            </div>
          </motion.div>
        </motion.div>
      )}
    </div>
  );
};