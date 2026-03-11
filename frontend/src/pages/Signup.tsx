import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { GlitchButton } from '../components/ui/GlitchButton';
import { useNavigate, Link } from 'react-router-dom';
import { User, Mail, Lock, Shield, Zap, Activity, UserPlus, ArrowRight, AlertCircle, Sparkles } from 'lucide-react';

import { createUserWithEmailAndPassword } from 'firebase/auth';
import { doc, setDoc, serverTimestamp } from 'firebase/firestore';
import { auth, db } from '../firebase';

/* =========================
   ANIMATED BACKGROUND
========================= */
const FloatingParticles = () => {
  const particles = Array.from({ length: 50 });
  
  return (
    <div className="absolute inset-0 overflow-hidden pointer-events-none">
      {particles.map((_, i) => {
        const size = Math.random() * 4 + 1;
        const duration = 15 + Math.random() * 15;
        const delay = Math.random() * 5;
        
        return (
          <motion.div
            key={i}
            className="absolute rounded-full"
            style={{
              width: size,
              height: size,
              left: `${Math.random() * 100}%`,
              top: `${Math.random() * 100}%`,
              background: i % 3 === 0 ? '#ff0033' : i % 3 === 1 ? '#ffffff' : '#00ffff',
              boxShadow: `0 0 ${size * 8}px currentColor`,
            }}
            animate={{
              x: [(Math.random() - 0.5) * 300, (Math.random() - 0.5) * 300],
              y: [(Math.random() - 0.5) * 300, (Math.random() - 0.5) * 300],
              opacity: [0.2, 0.8, 0.2],
              scale: [1, 1.5, 1],
            }}
            transition={{
              duration,
              repeat: Infinity,
              delay,
              ease: 'easeInOut',
            }}
          />
        );
      })}
    </div>
  );
};

const AmbientGlow = () => (
  <>
    <motion.div
      className="absolute w-[600px] h-[600px] rounded-full blur-[150px]"
      style={{
        background: 'radial-gradient(circle, rgba(255,0,51,0.4) 0%, transparent 70%)',
        top: '20%',
        right: '10%',
      }}
      animate={{
        scale: [1, 1.3, 1],
        opacity: [0.4, 0.7, 0.4],
        x: [0, -50, 0],
        y: [0, 30, 0],
      }}
      transition={{ duration: 10, repeat: Infinity, ease: 'easeInOut' }}
    />
    <motion.div
      className="absolute w-[500px] h-[500px] rounded-full blur-[140px]"
      style={{
        background: 'radial-gradient(circle, rgba(0,255,255,0.3) 0%, transparent 70%)',
        bottom: '15%',
        left: '10%',
      }}
      animate={{
        scale: [1.2, 1, 1.2],
        opacity: [0.3, 0.6, 0.3],
        x: [0, 40, 0],
        y: [0, -40, 0],
      }}
      transition={{ duration: 12, repeat: Infinity, ease: 'easeInOut', delay: 1 }}
    />
  </>
);

/* =========================
   ANIMATED VISUAL SIDE
========================= */
const AnimatedVisual = () => {
  const orbitNodes = [
    { size: 40, distance: 100, duration: 8, color: 'from-red-500 to-pink-500', icon: Shield, delay: 0 },
    { size: 35, distance: 130, duration: 10, color: 'from-cyan-400 to-blue-500', icon: Zap, delay: 0.5 },
    { size: 38, distance: 160, duration: 12, color: 'from-purple-500 to-red-500', icon: Activity, delay: 1 },
    { size: 33, distance: 110, duration: 9, color: 'from-pink-500 to-red-500', icon: UserPlus, delay: 1.5 },
  ];

  return (
    <div className="w-1/2 relative flex items-center justify-center px-8 bg-gradient-to-bl from-black via-gray-900 to-black">
      <FloatingParticles />
      <AmbientGlow />

      <div className="relative z-10 w-full max-w-[600px]">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: -30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.2 }}
          className="text-center mb-16"
        >
          <motion.div
            className="inline-flex items-center gap-3 mb-6 bg-red-500/10 backdrop-blur-sm border border-red-500/20 rounded-full px-6 py-3"
            animate={{
              boxShadow: [
                '0 0 20px rgba(255,0,51,0.2)',
                '0 0 40px rgba(255,0,51,0.4)',
                '0 0 20px rgba(255,0,51,0.2)',
              ],
            }}
            transition={{ duration: 3, repeat: Infinity }}
          >
            <motion.div
              animate={{ rotate: [0, 360] }}
              transition={{ duration: 3, repeat: Infinity, ease: 'linear' }}
            >
              <Sparkles className="text-red-400" size={18} />
            </motion.div>
            <span className="text-red-400 font-semibold text-sm tracking-wider uppercase">
              Join The Network
            </span>
          </motion.div>

          <h1
            className="font-bold mb-5 leading-none"
            style={{
              fontSize: '6rem',
              fontFamily: '"Playfair Display", Georgia, serif',
              background: 'linear-gradient(135deg, #ffffff 0%, #ff0033 50%, #00ffff 100%)',
              backgroundSize: '200% 100%',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
              backgroundClip: 'text',
              letterSpacing: '0.08em',
            }}
          >
            ChaosLens
          </h1>
          <p className="text-gray-300 text-lg max-w-xl mx-auto leading-relaxed" style={{ fontFamily: '"Crimson Text", Georgia, serif' }}>
            Enter a realm of classified operations
            <br />
            <span className="text-red-500/90 font-semibold italic">Your clearance awaits verification</span>
          </p>
        </motion.div>

        {/* Central Orbiting System */}
        <motion.div
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 1, delay: 0.5 }}
          className="relative h-[400px] flex items-center justify-center"
        >
          {/* Central core */}
          <motion.div
            className="relative w-28 h-28 rounded-full bg-gradient-to-br from-red-500 via-pink-500 to-purple-500 flex items-center justify-center"
            animate={{
              boxShadow: [
                '0 0 40px rgba(255,0,51,0.8), 0 0 80px rgba(255,0,51,0.4), 0 0 120px rgba(255,0,51,0.2)',
                '0 0 60px rgba(255,0,51,1), 0 0 100px rgba(255,0,51,0.6), 0 0 140px rgba(255,0,51,0.3)',
                '0 0 40px rgba(255,0,51,0.8), 0 0 80px rgba(255,0,51,0.4), 0 0 120px rgba(255,0,51,0.2)',
              ],
              scale: [1, 1.08, 1],
              rotate: [0, 180, 360],
            }}
            transition={{ duration: 4, repeat: Infinity, ease: 'easeInOut' }}
          >
            <motion.div
              animate={{ rotate: [0, -360] }}
              transition={{ duration: 8, repeat: Infinity, ease: 'linear' }}
            >
              <Shield className="text-white" size={40} />
            </motion.div>

            {/* Pulse rings */}
            {[...Array(4)].map((_, i) => (
              <motion.div
                key={i}
                className="absolute inset-0 rounded-full border-2 border-white/40"
                initial={{ scale: 1, opacity: 1 }}
                animate={{
                  scale: [1, 2.2, 2.8],
                  opacity: [1, 0.4, 0],
                }}
                transition={{
                  duration: 2.5,
                  repeat: Infinity,
                  delay: i * 0.6,
                  ease: 'easeOut',
                }}
              />
            ))}
          </motion.div>

          {/* Orbiting nodes */}
          {orbitNodes.map((node, i) => (
            <motion.div
              key={i}
              className="absolute"
              animate={{ rotate: 360 }}
              transition={{
                duration: node.duration,
                repeat: Infinity,
                ease: 'linear',
                delay: node.delay,
              }}
              style={{
                width: node.distance * 2,
                height: node.distance * 2,
              }}
            >
              <motion.div
                className={`absolute top-0 left-1/2 -translate-x-1/2 -translate-y-1/2 rounded-full bg-gradient-to-br ${node.color} flex items-center justify-center cursor-pointer`}
                style={{
                  width: node.size,
                  height: node.size,
                }}
                animate={{
                  boxShadow: [
                    '0 0 15px rgba(255,0,51,0.5), 0 0 30px rgba(255,0,51,0.3)',
                    '0 0 25px rgba(255,0,51,0.8), 0 0 50px rgba(255,0,51,0.5)',
                    '0 0 15px rgba(255,0,51,0.5), 0 0 30px rgba(255,0,51,0.3)',
                  ],
                  scale: [1, 1.15, 1],
                }}
                transition={{ duration: 2, repeat: Infinity, delay: i * 0.3 }}
                whileHover={{ scale: 1.3, rotate: 360, transition: { duration: 0.4 } }}
              >
                <motion.div
                  animate={{ rotate: [-360, 0] }}
                  transition={{ duration: node.duration, repeat: Infinity, ease: 'linear' }}
                >
                  <node.icon className="text-white" size={node.size * 0.45} />
                </motion.div>
              </motion.div>
            </motion.div>
          ))}

          {/* Connection lines */}
          <svg className="absolute inset-0 w-full h-full pointer-events-none">
            {orbitNodes.map((node, i) => (
              <motion.circle
                key={i}
                cx="50%"
                cy="50%"
                r={node.distance}
                fill="none"
                stroke="rgba(255,0,51,0.2)"
                strokeWidth="1.5"
                strokeDasharray="8 4"
                animate={{ strokeDashoffset: [0, i % 2 === 0 ? -100 : 100] }}
                transition={{ duration: node.duration, repeat: Infinity, ease: 'linear' }}
              />
            ))}
          </svg>
        </motion.div>

        {/* Stats */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 1.2 }}
          className="flex justify-center gap-8 mt-12"
        >
          {[
            { label: 'Active Agents', value: '10K+' },
            { label: 'Operations', value: '50K+' },
            { label: 'Sectors', value: '200+' },
          ].map((stat, i) => (
            <motion.div
              key={stat.label}
              className="text-center"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 1.3 + i * 0.1 }}
              whileHover={{ scale: 1.1, y: -5 }}
            >
              <motion.div
                className="text-2xl font-bold bg-gradient-to-r from-red-500 to-pink-500 bg-clip-text text-transparent mb-1"
                animate={{ scale: [1, 1.05, 1] }}
                transition={{ duration: 2, repeat: Infinity, delay: i * 0.3 }}
              >
                {stat.value}
              </motion.div>
              <div className="text-xs text-gray-400 uppercase tracking-wider">{stat.label}</div>
            </motion.div>
          ))}
        </motion.div>
      </div>
    </div>
  );
};

/* =========================
   MAIN SIGNUP COMPONENT
========================= */
export const Signup = () => {
  const navigate = useNavigate();

  const [sector, setSector] = useState('HAWKINS LAB');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const passwordRegex = /^(?=.*[a-z])(?=.*[A-Z])(?=.*[\W_]).{6,4096}$/;

  const handleSignup = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);

    if (!passwordRegex.test(password)) {
      setError('Password must be 6–4096 characters and include uppercase, lowercase, and a special character.');
      return;
    }

    try {
      setLoading(true);

      const userCredential = await createUserWithEmailAndPassword(auth, email, password);
      const user = userCredential.user;

      await setDoc(doc(db, 'users', user.uid), {
        uid: user.uid,
        email: user.email,
        sector,
        createdAt: serverTimestamp(),
      });

      await setDoc(doc(db, 'settings', user.uid), {
        uid: user.uid,
        overlay: true,
        audio: true,
        notifications: true,
        createdAt: serverTimestamp(),
      });

      navigate('/');
    } catch (err: any) {
      if (err.code === 'auth/email-already-in-use') {
        setError('This email is already registered.');
      } else if (err.code === 'auth/invalid-email') {
        setError('Invalid email address.');
      } else if (err.code === 'auth/weak-password') {
        setError('Password is too weak.');
      } else {
        setError('Registration failed. Please try again.');
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen w-full flex relative overflow-hidden bg-black">
      {/* Animated grid */}
      <motion.div
        className="absolute inset-0 opacity-[0.05]"
        style={{
          backgroundImage:
            'linear-gradient(rgba(255, 0, 51, 0.4) 1px, transparent 1px), linear-gradient(90deg, rgba(255, 0, 51, 0.4) 1px, transparent 1px)',
          backgroundSize: '50px 50px',
        }}
        animate={{ backgroundPosition: ['0px 0px', '50px 50px'] }}
        transition={{ duration: 20, repeat: Infinity, ease: 'linear' }}
      />

      {/* LEFT SIDE - Form */}
      <div className="w-1/2 h-screen flex items-center justify-center px-20 relative z-10 bg-gradient-to-br from-black via-gray-900 to-black">
        <motion.div
          initial={{ opacity: 0, x: -60 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 1, delay: 0.3 }}
          className="w-full max-w-[520px]"
        >
          {/* Header */}
          <div className="mb-10">
            <motion.div
              initial={{ opacity: 0, y: -20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.5 }}
              className="flex items-center gap-4 mb-7"
            >
              <motion.div
                animate={{ rotate: [0, 10, -10, 0], scale: [1, 1.05, 1] }}
                transition={{ duration: 4, repeat: Infinity }}
                className="p-3 bg-red-500/10 rounded-xl border border-red-500/30"
              >
                <UserPlus className="text-red-500 w-10 h-10" />
              </motion.div>
              <div>
                <h1
                  className="font-bold tracking-tight leading-none mb-2"
                  style={{ fontSize: '3rem', fontFamily: '"Playfair Display", Georgia, serif', color: '#ffffff' }}
                >
                  Account Registration
                </h1>
                <p className="text-gray-500 text-[11px] tracking-[0.25em] uppercase font-semibold" style={{ fontFamily: '"JetBrains Mono", monospace' }}>
                  ACCOUNT CREATION
                </p>
              </div>
            </motion.div>

            <motion.p
              className="text-gray-400 leading-relaxed"
              style={{ fontFamily: '"Crimson Text", Georgia, serif', fontSize: '1.05rem' }}
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.7 }}
            >
              Create your account on the{' '}
              <span className="text-red-500 font-semibold italic">secure deepfake detection platform</span>.
              All credentials are encrypted.
            </motion.p>
          </div>

          {/* Form */}
          <form onSubmit={handleSignup} className="space-y-6">
            {/* Detection Mode */}
            <motion.div className="space-y-2.5" initial={{ opacity: 0, x: 30 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.9 }}>
              <label className="text-red-500 uppercase flex items-center gap-2.5 text-[11px] tracking-[0.2em] font-semibold" style={{ fontFamily: '"JetBrains Mono", monospace' }}>
                <Shield size={12} className="animate-pulse" />
                Detection Mode
              </label>
              <div className="relative group">
                <select
                  value={sector}
                  onChange={(e) => setSector(e.target.value)}
                  className="w-full bg-black/60 border-2 border-white/10 focus:border-red-500/60 focus:bg-black/80 text-white px-5 py-4 outline-none transition-all duration-500 rounded-xl appearance-none cursor-pointer"
                  style={{ fontFamily: '"JetBrains Mono", monospace', fontSize: '0.95rem' }}
                >
                  <option>ADVANCED DETECTION</option>
                  <option>REAL-TIME ANALYSIS</option>
                  <option>DEEP SCAN</option>
                </select>
              </div>
            </motion.div>

            {/* Email */}
            <motion.div className="space-y-2.5" initial={{ opacity: 0, x: 30 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 1 }}>
              <label className="text-red-500 uppercase flex items-center gap-2.5 text-[11px] tracking-[0.2em] font-semibold" style={{ fontFamily: '"JetBrains Mono", monospace' }}>
                <Mail size={12} className="animate-pulse" />
                Email Address
              </label>
              <div className="relative group">
                <Mail className="absolute left-5 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-600 group-focus-within:text-cyan-400 transition-all duration-300 z-10" />
                <input
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  required
                  className="w-full bg-black/60 border-2 border-white/10 focus:border-cyan-400/60 focus:bg-black/80 text-white pl-14 pr-5 py-4 outline-none transition-all duration-500 rounded-xl"
                  style={{ fontFamily: '"JetBrains Mono", monospace', fontSize: '0.95rem' }}
                  placeholder="Enter your email address"
                />
              </div>
            </motion.div>

            {/* Password */}
            <motion.div className="space-y-2.5" initial={{ opacity: 0, x: 30 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 1.1 }}>
              <label className="text-red-500 uppercase flex items-center gap-2.5 text-[11px] tracking-[0.2em] font-semibold" style={{ fontFamily: '"JetBrains Mono", monospace' }}>
                <Lock size={12} className="animate-pulse" />
                Password
              </label>
              <div className="relative group">
                <Lock className="absolute left-5 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-600 group-focus-within:text-red-500 transition-all duration-300 z-10" />
                <input
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  required
                  className="w-full bg-black/60 border-2 border-white/10 focus:border-red-500/60 focus:bg-black/80 text-white pl-14 pr-5 py-4 outline-none transition-all duration-500 rounded-xl"
                  style={{ fontFamily: '"JetBrains Mono", monospace', fontSize: '0.95rem' }}
                  placeholder="Create a strong password"
                />
              </div>
              <p className="text-xs text-gray-500 font-mono">Min 6 chars, uppercase, lowercase, special char</p>
            </motion.div>

            {/* Error */}
            {error && (
              <motion.div
                initial={{ opacity: 0, y: -15, scale: 0.95 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                className="bg-red-500/10 border-l-4 border-red-500 rounded-xl px-5 py-4 backdrop-blur-sm"
              >
                <p className="text-red-500 flex items-center gap-3" style={{ fontSize: '0.9rem', fontFamily: '"JetBrains Mono", monospace' }}>
                  <AlertCircle size={20} className="flex-shrink-0" />
                  <span>{error}</span>
                </p>
              </motion.div>
            )}

            {/* Submit */}
            <motion.div className="pt-5 flex gap-4" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 1.2 }}>
              <Link to="/login" className="flex-1">
                <button
                  type="button"
                  className="w-full bg-black/50 border-2 border-white/10 hover:border-white/30 text-white py-4 rounded-xl font-semibold tracking-wider transition-all duration-300"
                  style={{ fontFamily: '"JetBrains Mono", monospace', fontSize: '0.95rem' }}
                >
                  CANCEL
                </button>
              </Link>

              <GlitchButton className="flex-1 relative group overflow-hidden" disabled={loading}>
                <motion.div
                  className="absolute inset-0 bg-gradient-to-r from-red-500/0 via-red-500/40 to-red-500/0"
                  animate={{ x: ['-100%', '200%'] }}
                  transition={{ duration: 2.5, repeat: Infinity, ease: 'linear' }}
                />
                <span className="relative z-10 flex items-center justify-center gap-3.5 font-semibold tracking-wider" style={{ fontSize: '1rem', fontFamily: '"JetBrains Mono", monospace' }}>
                  {loading ? (
                    <>
                      <motion.div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full" animate={{ rotate: 360 }} transition={{ duration: 1, repeat: Infinity, ease: 'linear' }} />
                      REGISTERING...
                    </>
                  ) : (
                    <>
                      <UserPlus size={22} className="group-hover:scale-110 transition-transform duration-300" />
                      REGISTER
                      <ArrowRight size={22} className="group-hover:translate-x-1 transition-transform duration-300" />
                    </>
                  )}
                </span>
              </GlitchButton>
            </motion.div>

            {/* Footer */}
            <motion.div className="pt-8 flex justify-between items-center" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 1.4 }}>
              <Link to="/login" className="text-gray-600 hover:text-red-500 transition-all duration-500 flex items-center gap-2 group" style={{ fontSize: '0.85rem', fontFamily: '"JetBrains Mono", monospace' }}>
                <motion.span className="text-gray-700" whileHover={{ x: -4 }} transition={{ duration: 0.3 }}>
                  [
                </motion.span>
                <span className="group-hover:tracking-widest transition-all duration-500 font-medium">ALREADY HAVE AN ACCOUNT?</span>
                <motion.span className="text-gray-700" whileHover={{ x: 4 }} transition={{ duration: 0.3 }}>
                  ]
                </motion.span>
              </Link>

              <motion.div className="flex gap-2.5" animate={{ opacity: [0.4, 0.8, 0.4] }} transition={{ duration: 4, repeat: Infinity }}>
                <div className="w-2.5 h-2.5 bg-red-500 rounded-full shadow-[0_0_10px_rgba(255,0,51,0.6)]" />
                <div className="w-2.5 h-2.5 bg-white/30 rounded-full" />
                <div className="w-2.5 h-2.5 bg-white/30 rounded-full" />
              </motion.div>
            </motion.div>
          </form>
        </motion.div>
      </div>

      {/* RIGHT SIDE - Animated Visual */}
      <AnimatedVisual />
    </div>
  );
};