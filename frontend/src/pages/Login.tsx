import React, { useState, useEffect, useRef } from 'react';
import { motion, useSpring } from 'framer-motion';
import { GlitchButton } from '../components/ui/GlitchButton';
import { useNavigate, Link } from 'react-router-dom';
import { Lock, User, Eye, Shield, Activity, Zap, AlertCircle } from 'lucide-react';
import { signInWithEmailAndPassword } from "firebase/auth";
import { auth } from "../firebase";

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
        left: '10%',
      }}
      animate={{
        scale: [1, 1.3, 1],
        opacity: [0.4, 0.7, 0.4],
        x: [0, 50, 0],
        y: [0, -30, 0],
      }}
      transition={{ duration: 10, repeat: Infinity, ease: 'easeInOut' }}
    />
    <motion.div
      className="absolute w-[500px] h-[500px] rounded-full blur-[140px]"
      style={{
        background: 'radial-gradient(circle, rgba(0,255,255,0.3) 0%, transparent 70%)',
        bottom: '15%',
        right: '10%',
      }}
      animate={{
        scale: [1.2, 1, 1.2],
        opacity: [0.3, 0.6, 0.3],
        x: [0, -40, 0],
        y: [0, 40, 0],
      }}
      transition={{ duration: 12, repeat: Infinity, ease: 'easeInOut', delay: 1 }}
    />
  </>
);

/* =========================
   REALISTIC WATCHING EYE
========================= */
interface EyeProps {
  passwordFocused: boolean;
  emailFocused: boolean;
  eyeX: any;
  eyeY: any;
  blinkState: boolean;
}

const RealisticEye = ({ passwordFocused, emailFocused, eyeX, eyeY, blinkState }: EyeProps) => {
  const isActive = passwordFocused || emailFocused;

  return (
    <motion.div
      className="w-72 h-72 relative"
      style={{ perspective: '1500px' }}
      initial={{ opacity: 0, scale: 0.8 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 1 }}
    >
      {/* Outer glow */}
      <motion.div
        className="absolute inset-0 rounded-full blur-3xl"
        style={{
          background: passwordFocused
            ? 'radial-gradient(circle, rgba(255,0,51,0.6) 0%, transparent 70%)'
            : emailFocused
            ? 'radial-gradient(circle, rgba(0,255,255,0.5) 0%, transparent 70%)'
            : 'radial-gradient(circle, rgba(255,255,255,0.3) 0%, transparent 70%)',
        }}
        animate={{
          scale: isActive ? [1, 1.4, 1] : [1, 1.2, 1],
          opacity: isActive ? [0.6, 1, 0.6] : [0.4, 0.7, 0.4],
        }}
        transition={{ duration: 3, repeat: Infinity }}
      />

      {/* Eye socket */}
      <motion.div
        className="absolute inset-0 rounded-full flex items-center justify-center"
        style={{
          background: 'radial-gradient(circle at 35% 35%, #2a2a2a, #0a0a0a 70%)',
          boxShadow: passwordFocused
            ? 'inset 0 0 80px rgba(0,0,0,1), inset 0 15px 40px rgba(255,0,51,0.5), 0 30px 90px rgba(255,0,51,0.7), 0 0 130px rgba(255,0,51,0.5)'
            : emailFocused
            ? 'inset 0 0 80px rgba(0,0,0,1), inset 0 15px 40px rgba(0,255,255,0.4), 0 30px 90px rgba(0,255,255,0.6), 0 0 120px rgba(0,255,255,0.4)'
            : 'inset 0 0 80px rgba(0,0,0,1), inset 0 15px 40px rgba(0,0,0,0.7), 0 30px 90px rgba(0,0,0,0.9), 0 0 100px rgba(255,255,255,0.2)',
        }}
        animate={{
          rotateX: isActive ? [0, 3, -3, 0] : 0,
        }}
        transition={{ duration: 4, repeat: Infinity }}
      >
        {/* Decorative rings */}
        <motion.div
          className="absolute inset-8 rounded-full border-2"
          style={{
            borderColor: passwordFocused
              ? 'rgba(255,0,51,0.5)'
              : emailFocused
              ? 'rgba(0,255,255,0.4)'
              : 'rgba(255,255,255,0.25)',
          }}
          animate={{ rotate: 360 }}
          transition={{ duration: 25, repeat: Infinity, ease: 'linear' }}
        />
        <motion.div
          className="absolute inset-12 rounded-full border"
          style={{
            borderColor: passwordFocused
              ? 'rgba(255,0,51,0.3)'
              : emailFocused
              ? 'rgba(0,255,255,0.25)'
              : 'rgba(255,255,255,0.15)',
          }}
          animate={{ rotate: -360 }}
          transition={{ duration: 20, repeat: Infinity, ease: 'linear' }}
        />

        {!blinkState && !passwordFocused ? (
          <>
            {/* Eyeball */}
            <motion.div
              style={{ x: eyeX, y: eyeY }}
              className="relative w-36 h-36 rounded-full flex items-center justify-center"
              animate={{ scale: isActive ? [1, 1.05, 1] : 1 }}
              transition={{ duration: 2, repeat: Infinity }}
            >
              {/* Sclera (white part) */}
              <div
                className="absolute inset-0 rounded-full"
                style={{
                  background: 'radial-gradient(circle at 45% 40%, #f0f0f0, #f8f8f8 50%, #ffffff)',
                  boxShadow: 'inset 0 -10px 25px rgba(0,0,0,0.25), 0 0 60px rgba(255,255,255,0.6)',
                }}
              />

              {/* Iris - outer layer */}
              <motion.div
                className="absolute inset-0 rounded-full opacity-75"
                style={{
                  background: emailFocused
                    ? 'radial-gradient(circle at 45% 45%, rgba(0,255,255,0.7) 0%, rgba(0,255,255,0.5) 45%, transparent 75%)'
                    : 'radial-gradient(circle at 45% 45%, rgba(100,180,255,0.6) 0%, rgba(80,160,240,0.4) 45%, transparent 75%)',
                }}
                animate={{ rotate: -360 }}
                transition={{ duration: 35, repeat: Infinity, ease: 'linear' }}
              />

              {/* Iris - inner layer */}
              <motion.div
                className="absolute inset-6 rounded-full opacity-65"
                style={{
                  background: emailFocused
                    ? 'radial-gradient(circle at 55% 35%, rgba(0,139,139,0.5) 0%, transparent 70%)'
                    : 'radial-gradient(circle at 55% 35%, rgba(0,100,200,0.5) 0%, transparent 70%)',
                }}
                animate={{ rotate: 360 }}
                transition={{ duration: 30, repeat: Infinity, ease: 'linear' }}
              />

              {/* Iris fibers/striation */}
              {[...Array(20)].map((_, i) => (
                <div
                  key={i}
                  className="absolute w-full h-[2px] top-1/2"
                  style={{
                    background: emailFocused
                      ? 'linear-gradient(to right, transparent, rgba(0,255,255,0.4), transparent)'
                      : 'linear-gradient(to right, transparent, rgba(100,150,200,0.35), transparent)',
                    transform: `rotate(${i * 18}deg)`,
                    transformOrigin: 'center',
                  }}
                />
              ))}

              {/* Pupil */}
              <motion.div
                className="relative w-16 h-16 bg-black rounded-full overflow-hidden"
                style={{
                  boxShadow: emailFocused
                    ? '0 0 30px rgba(0,255,255,0.7), inset 0 3px 12px rgba(0,0,0,1)'
                    : '0 0 30px rgba(0,0,0,0.9), inset 0 3px 12px rgba(0,0,0,1)',
                }}
                animate={{
                  scale: emailFocused ? [1, 0.8, 1] : 1,
                }}
                transition={{ duration: 3, repeat: Infinity }}
              >
                {/* Pupil highlights */}
                <div className="absolute top-2 left-2 w-4 h-4 bg-white/90 rounded-full blur-[0.5px]" />
                <div className="absolute bottom-2.5 right-2.5 w-2 h-2 bg-white/50 rounded-full blur-[0.5px]" />
              </motion.div>
            </motion.div>

            {/* Light reflections */}
            <motion.div
              style={{ x: eyeX, y: eyeY }}
              className="absolute top-8 left-8 w-9 h-9 bg-white/85 rounded-full blur-sm pointer-events-none"
            />
            <motion.div
              style={{ x: eyeX, y: eyeY }}
              className="absolute top-10 left-10 w-6 h-6 bg-white/65 rounded-full blur-[3px] pointer-events-none"
            />
          </>
        ) : (
          // Closed eye animation - for both blink and password focus
          <motion.div
            className="relative w-44 h-8 flex items-center justify-center"
            initial={{ scaleY: 1 }}
            animate={{ scaleY: passwordFocused ? 0.08 : 0.05 }}
            transition={{ 
              duration: passwordFocused ? 0.6 : 0.12,
              ease: passwordFocused ? [0.65, 0, 0.35, 1] : 'easeInOut'
            }}
          >
            {/* Main eyelid */}
            <div 
              className="w-full h-full rounded-full shadow-2xl relative overflow-hidden"
              style={{
                background: passwordFocused 
                  ? 'linear-gradient(to bottom, #8b0000 0%, #4a0000 50%, #2a0000 100%)'
                  : 'linear-gradient(to bottom, #6b6b6b 0%, #4a4a4a 50%, #2a2a2a 100%)',
                boxShadow: passwordFocused
                  ? '0 0 40px rgba(255,0,51,0.8), inset 0 2px 8px rgba(0,0,0,0.8), inset 0 -2px 8px rgba(139,0,0,0.5)'
                  : '0 0 20px rgba(0,0,0,0.5), inset 0 2px 8px rgba(0,0,0,0.8)',
              }}
            >
              {/* Eyelid texture/shine */}
              <div 
                className="absolute inset-0 rounded-full"
                style={{
                  background: passwordFocused
                    ? 'linear-gradient(135deg, rgba(255,0,51,0.3) 0%, transparent 50%, rgba(139,0,0,0.2) 100%)'
                    : 'linear-gradient(135deg, rgba(255,255,255,0.15) 0%, transparent 50%, rgba(0,0,0,0.2) 100%)',
                }}
              />
              
              {/* Intense red glow effect for password focus */}
              {passwordFocused && (
                <motion.div
                  className="absolute inset-0 rounded-full"
                  style={{
                    background: 'radial-gradient(ellipse at center, rgba(255,0,51,0.4) 0%, transparent 70%)',
                  }}
                  animate={{
                    opacity: [0.4, 0.8, 0.4],
                  }}
                  transition={{
                    duration: 1.8,
                    repeat: Infinity,
                    ease: 'easeInOut',
                  }}
                />
              )}
            </div>

            {/* Eyelid crease/detail */}
            <motion.div
              className="absolute top-0 w-full h-[2px] rounded-full"
              style={{
                background: passwordFocused
                  ? 'linear-gradient(to right, transparent, rgba(255,0,51,0.6) 50%, transparent)'
                  : 'linear-gradient(to right, transparent, rgba(100,100,100,0.4) 50%, transparent)',
                boxShadow: passwordFocused 
                  ? '0 0 15px rgba(255,0,51,0.8)'
                  : '0 0 8px rgba(0,0,0,0.3)',
              }}
              animate={passwordFocused ? {
                opacity: [0.6, 1, 0.6],
              } : {}}
              transition={{
                duration: 2,
                repeat: Infinity,
              }}
            />
          </motion.div>
        )}
      </motion.div>

      {/* Eyelashes - hide when eyes are closed */}
      {!passwordFocused && (
        <div className="absolute inset-0 rounded-full overflow-hidden pointer-events-none">
          {[...Array(15)].map((_, i) => (
            <motion.div
              key={i}
              className="absolute w-[3px] h-14 bg-gradient-to-b from-gray-900 to-transparent origin-bottom"
              style={{
                top: '0%',
                left: `${8 + i * 6}%`,
                transform: `rotate(${-45 + i * 7}deg)`,
              }}
              animate={{ opacity: blinkState ? 0 : [0.3, 0.7, 0.3] }}
              transition={{ duration: 3, repeat: Infinity, delay: i * 0.1 }}
            />
          ))}
        </div>
      )}
    </motion.div>
  );
};

/* =========================
   MAIN LOGIN COMPONENT
========================= */
export const Login = () => {
  const navigate = useNavigate();
  const eyeContainerRef = useRef<HTMLDivElement>(null);

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [passwordFocused, setPasswordFocused] = useState(false);
  const [emailFocused, setEmailFocused] = useState(false);
  const [blinkState, setBlinkState] = useState(false);

  // Eye tracking
  const springConfig = { damping: 20, stiffness: 150, mass: 0.5 };
  const eyeX = useSpring(0, springConfig);
  const eyeY = useSpring(0, springConfig);

  useEffect(() => {
    const handleMouse = (e: MouseEvent) => {
      if (eyeContainerRef.current) {
        const rect = eyeContainerRef.current.getBoundingClientRect();
        const centerX = rect.left + rect.width / 2;
        const centerY = rect.top + rect.height / 2;
        const deltaX = e.clientX - centerX;
        const deltaY = e.clientY - centerY;
        const distance = Math.sqrt(deltaX * deltaX + deltaY * deltaY);
        const angle = Math.atan2(deltaY, deltaX);
        const maxMovement = 30;
        const normalizedDistance = Math.min(distance / 250, 1);
        const moveX = Math.cos(angle) * normalizedDistance * maxMovement;
        const moveY = Math.sin(angle) * normalizedDistance * maxMovement;
        eyeX.set(moveX);
        eyeY.set(moveY);
      }
    };

    window.addEventListener('mousemove', handleMouse);
    return () => window.removeEventListener('mousemove', handleMouse);
  }, [eyeX, eyeY]);

  // Blinking
  useEffect(() => {
    if (passwordFocused) return;
    const interval = setInterval(() => {
      if (!emailFocused && Math.random() > 0.4) {
        setBlinkState(true);
        setTimeout(() => setBlinkState(false), 100 + Math.random() * 80);
      }
    }, 2000 + Math.random() * 4000);
    return () => clearInterval(interval);
  }, [passwordFocused, emailFocused]);

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
  
    try {
      await signInWithEmailAndPassword(auth, email, password);
      navigate("/home");
    } catch (err: any) {
      console.error("Firebase Auth Error:", err.code);
  
      switch (err.code) {
        case "auth/invalid-email":
          setError("Invalid email address.");
          break;
  
        case "auth/invalid-credential":
          setError("Email not registered or password is incorrect.");
          break;
  
        case "auth/user-disabled":
          setError("This account has been disabled.");
          break;
  
        case "auth/too-many-requests":
          setError("Too many failed attempts. Please try again later.");
          break;
  
        default:
          setError("Authentication failed. Please try again.");
      }
    } finally {
      setLoading(false);
    }
  };

  const isActive = passwordFocused || emailFocused;

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

      {/* LEFT SIDE - Eyes */}
      <div className="w-1/2 h-screen flex items-center justify-center relative bg-gradient-to-br from-black via-gray-900 to-black">
        <FloatingParticles />
        <AmbientGlow />

        <div ref={eyeContainerRef} className="relative z-10 w-full max-w-[700px] px-10">
          {/* Header */}
          <motion.div
            initial={{ opacity: 0, y: -30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="text-center mb-8"
          >
            <h1
              className="font-bold mb-5 leading-none"
              style={{
                fontSize: '5.5rem',
                fontFamily: '"Playfair Display", Georgia, serif',
                background: 'linear-gradient(135deg, #ffffff 0%, #ff0033 50%, #ffffff 100%)',
                backgroundSize: '200% 100%',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
                backgroundClip: 'text',
                letterSpacing: '0.08em',
              }}
            >
              ChaosLens
            </h1>
            <p className="text-gray-300 text-xl max-w-xl mx-auto leading-relaxed" style={{ fontFamily: '"Crimson Text", Georgia, serif' }}>
              Advanced deepfake detection platform
              <br />
              <span className="text-red-500/90 font-semibold italic">Secure authentication required</span>
            </p>
          </motion.div>

          {/* Status badge - moved below header */}
          <motion.div
            className="flex items-center justify-center gap-3 bg-black/70 backdrop-blur-lg border border-white/20 rounded-full px-6 py-3 mb-12 mx-auto w-fit"
            initial={{ opacity: 0, y: -20 }}
            animate={{
              opacity: 1,
              y: 0,
              boxShadow: passwordFocused
                ? ['0 0 20px rgba(255,0,51,0.4)', '0 0 50px rgba(255,0,51,0.7)', '0 0 20px rgba(255,0,51,0.4)']
                : emailFocused
                ? ['0 0 20px rgba(0,255,255,0.3)', '0 0 40px rgba(0,255,255,0.6)', '0 0 20px rgba(0,255,255,0.3)']
                : '0 0 25px rgba(255,255,255,0.15)',
            }}
            transition={{ boxShadow: { duration: 2, repeat: Infinity }, opacity: { duration: 0.6 } }}
          >
            <Activity className={passwordFocused ? 'text-red-500' : emailFocused ? 'text-cyan-400' : 'text-white/70'} size={16} />
            <span className={`text-sm font-mono tracking-[0.15em] font-medium ${passwordFocused ? 'text-red-500' : emailFocused ? 'text-cyan-400' : 'text-white/70'}`}>
              {passwordFocused ? 'AUTHENTICATION' : emailFocused ? 'EMAIL VERIFICATION' : 'SYSTEM READY'}
            </span>
            <motion.div
              className={`w-2.5 h-2.5 rounded-full ${passwordFocused ? 'bg-red-500' : emailFocused ? 'bg-cyan-400' : 'bg-white/70'}`}
              animate={{ scale: [1, 1.5, 1], opacity: [0.6, 1, 0.6] }}
              transition={{ duration: 1.5, repeat: Infinity }}
            />
          </motion.div>

          {/* Eyes */}
          <div className="flex items-center justify-center gap-12 mb-20">
            <RealisticEye
              passwordFocused={passwordFocused}
              emailFocused={emailFocused}
              eyeX={eyeX}
              eyeY={eyeY}
              blinkState={blinkState}
            />
            <RealisticEye
              passwordFocused={passwordFocused}
              emailFocused={emailFocused}
              eyeX={eyeX}
              eyeY={eyeY}
              blinkState={blinkState}
            />
          </div>

          {/* Status indicators */}
          <motion.div
            className="flex justify-center gap-4"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.8 }}
          >
            {[
              { icon: Eye, label: 'TRACKING', active: true },
              { icon: Shield, label: 'SECURED', active: isActive },
              { icon: Zap, label: 'ANALYZING', active: passwordFocused },
            ].map((item, i) => (
              <motion.div
                key={item.label}
                className="flex items-center gap-2.5 bg-black/50 backdrop-blur-md border rounded-xl px-4 py-2.5"
                style={{ borderColor: item.active ? 'rgba(255, 0, 51, 0.4)' : 'rgba(255, 255, 255, 0.15)' }}
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: 1 + i * 0.15 }}
              >
                <motion.div animate={{ scale: item.active ? [1, 1.3, 1] : 1, opacity: item.active ? [0.7, 1, 0.7] : 0.5 }} transition={{ duration: 2.5, repeat: Infinity }}>
                  <item.icon className={item.active ? 'text-red-500' : 'text-white/50'} size={15} />
                </motion.div>
                <span className={`text-xs font-mono tracking-[0.12em] font-medium ${item.active ? 'text-red-500' : 'text-white/50'}`}>{item.label}</span>
              </motion.div>
            ))}
          </motion.div>
        </div>
      </div>

      {/* RIGHT SIDE - Form */}
      <div className="w-1/2 h-screen flex items-center justify-center px-20 relative z-10 bg-gradient-to-bl from-black via-gray-900 to-black">
        <motion.div
          initial={{ opacity: 0, x: 60 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 1, delay: 0.3 }}
          className="w-full max-w-[520px]"
        >
          {/* Header */}
          <div className="mb-12">
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
                <Shield className="text-red-500 w-10 h-10" />
              </motion.div>
              <div>
                <h1
                  className="font-bold tracking-tight leading-none mb-2"
                  style={{ fontSize: '3rem', fontFamily: '"Playfair Display", Georgia, serif', color: '#ffffff' }}
                >
                  Welcome Back
                </h1>
                <p className="text-gray-500 text-[11px] tracking-[0.25em] uppercase font-semibold" style={{ fontFamily: '"JetBrains Mono", monospace' }}>
                  AUTHORIZED ACCESS
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
              Your credentials are being verified through our{' '}
              <span className="text-red-500 font-semibold italic">biometric system</span>.
              Please proceed with authentication.
            </motion.p>
          </div>

          {/* Form */}
          <form onSubmit={handleLogin} className="space-y-6">
            {/* Email */}
            <motion.div className="space-y-2.5" initial={{ opacity: 0, x: 30 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.8 }}>
              <label className="text-red-500 uppercase flex items-center gap-2.5 text-[11px] tracking-[0.2em] font-semibold" style={{ fontFamily: '"JetBrains Mono", monospace' }}>
                <Zap size={12} className="animate-pulse" />
                Email Address
              </label>
              <div className="relative group">
                <User className="absolute left-5 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-600 group-focus-within:text-cyan-400 transition-all duration-300 z-10" />
                <input
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  onFocus={() => setEmailFocused(true)}
                  onBlur={() => setEmailFocused(false)}
                  required
                  className="w-full bg-black/60 border-2 border-white/10 focus:border-cyan-400/60 focus:bg-black/80 text-white pl-14 pr-5 py-4 outline-none transition-all duration-500 rounded-xl"
                  style={{ fontFamily: '"JetBrains Mono", monospace', fontSize: '0.95rem' }}
                  placeholder="Enter your email address"
                />
                <motion.div
                  className="absolute inset-0 border-2 border-cyan-400/0 group-focus-within:border-cyan-400/40 pointer-events-none transition-all duration-500 rounded-xl"
                  animate={{ boxShadow: emailFocused ? ['0 0 0px rgba(0,255,255,0)', '0 0 30px rgba(0,255,255,0.4)', '0 0 0px rgba(0,255,255,0)'] : '0 0 0px rgba(0,255,255,0)' }}
                  transition={{ duration: 2.5, repeat: Infinity }}
                />
              </div>
            </motion.div>

            {/* Password */}
            <motion.div className="space-y-2.5" initial={{ opacity: 0, x: 30 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.9 }}>
              <label className="text-red-500 uppercase flex items-center gap-2.5 text-[11px] tracking-[0.2em] font-semibold" style={{ fontFamily: '"JetBrains Mono", monospace' }}>
                <Shield size={12} className="animate-pulse" />
                Password
              </label>
              <div className="relative group">
                <Lock className="absolute left-5 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-600 group-focus-within:text-red-500 transition-all duration-300 z-10" />
                <input
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  onFocus={() => setPasswordFocused(true)}
                  onBlur={() => setPasswordFocused(false)}
                  required
                  className="w-full bg-black/60 border-2 border-white/10 focus:border-red-500/60 focus:bg-black/80 text-white pl-14 pr-5 py-4 outline-none transition-all duration-500 rounded-xl"
                  style={{ fontFamily: '"JetBrains Mono", monospace', fontSize: '0.95rem' }}
                  placeholder="Enter your password"
                />
                <motion.div
                  className="absolute inset-0 border-2 border-red-500/0 group-focus-within:border-red-500/40 pointer-events-none transition-all duration-500 rounded-xl"
                  animate={{ boxShadow: passwordFocused ? ['0 0 0px rgba(255,0,51,0)', '0 0 35px rgba(255,0,51,0.5)', '0 0 0px rgba(255,0,51,0)'] : '0 0 0px rgba(255,0,51,0)' }}
                  transition={{ duration: 2, repeat: Infinity }}
                />
              </div>
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
            <motion.div className="pt-5" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 1 }}>
              <GlitchButton className="w-full relative group overflow-hidden" disabled={loading}>
                <motion.div
                  className="absolute inset-0 bg-gradient-to-r from-red-500/0 via-red-500/40 to-red-500/0"
                  animate={{ x: ['-100%', '200%'] }}
                  transition={{ duration: 2.5, repeat: Infinity, ease: 'linear' }}
                />
                <span className="relative z-10 flex items-center justify-center gap-3.5 font-semibold tracking-wider" style={{ fontSize: '1rem', fontFamily: '"JetBrains Mono", monospace' }}>
                  {loading ? (
                    <>
                      <motion.div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full" animate={{ rotate: 360 }} transition={{ duration: 1, repeat: Infinity, ease: 'linear' }} />
                      AUTHENTICATING...
                    </>
                  ) : (
                    <>
                      <Eye size={22} className="group-hover:scale-110 transition-transform duration-300" />
                      LOGIN
                    </>
                  )}
                </span>
              </GlitchButton>
            </motion.div>

            {/* Footer */}
            <motion.div className="pt-8 flex justify-between items-center" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 1.2 }}>
              <Link to="/signup" className="text-gray-600 hover:text-red-500 transition-all duration-500 flex items-center gap-2 group" style={{ fontSize: '0.85rem', fontFamily: '"JetBrains Mono", monospace' }}>
                <motion.span className="text-gray-700" whileHover={{ x: -4 }} transition={{ duration: 0.3 }}>
                  [
                </motion.span>
                <span className="group-hover:tracking-widest transition-all duration-500 font-medium">CREATE ACCOUNT</span>
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
    </div>
  );
};