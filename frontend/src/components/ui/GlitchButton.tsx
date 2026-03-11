import React from 'react';
import { motion } from 'framer-motion';
import { clsx } from 'clsx';
import { useAudio } from '../../context/AudioContext'; // Correct relative path

interface GlitchButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'danger' | 'ghost';
  children: React.ReactNode;
}

export const GlitchButton: React.FC<GlitchButtonProps> = ({ 
  variant = 'primary', 
  children, 
  className,
  onClick,
  ...props 
}) => {
  const { playClick } = useAudio(); // This will play click sound if audio enabled

  const handleClick = (e: React.MouseEvent<HTMLButtonElement>) => {
    playClick(); // Play sound
    if (onClick) onClick(e); // call original onClick
  };

  const baseStyles = "relative px-6 py-3 font-display font-bold uppercase tracking-wider transition-all duration-200 overflow-hidden group";
  
  const variants = {
    primary: "bg-transparent border border-blood text-blood hover:bg-blood hover:text-black shadow-[0_0_10px_rgba(255,0,51,0.3)] hover:shadow-[0_0_20px_rgba(255,0,51,0.6)]",
    danger: "bg-transparent border border-void text-void hover:bg-void hover:text-white shadow-[0_0_10px_rgba(107,33,168,0.3)] hover:shadow-[0_0_20px_rgba(107,33,168,0.6)]",
    ghost: "text-gray-400 hover:text-neon border border-transparent hover:border-neon/30"
  };

  return (
    <motion.button
      whileHover={{ scale: 1.02 }}
      whileTap={{ scale: 0.98 }}
      className={clsx(baseStyles, variants[variant], className)}
      onClick={handleClick} // Use our wrapper
      {...props}
    >
      <span className="relative z-10 flex items-center justify-center gap-2">
        {children}
      </span>
      
      {/* Glitch Overlay Effect on Hover */}
      <div className="absolute inset-0 bg-white/20 translate-x-[-100%] group-hover:translate-x-[100%] transition-transform duration-300 skew-x-12" />
    </motion.button>
  );
};
