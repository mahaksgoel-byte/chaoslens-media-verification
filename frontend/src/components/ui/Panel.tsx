import React from 'react';
import { clsx } from 'clsx';

interface PanelProps {
  children: React.ReactNode;
  className?: string;
  title?: string;
  glow?: 'red' | 'purple' | 'cyan';
}

export const Panel: React.FC<PanelProps> = ({ children, className, title, glow = 'red' }) => {
  const glowStyles = {
    red: "border-blood/30 shadow-[0_0_15px_rgba(255,0,51,0.1)]",
    purple: "border-void/50 shadow-[0_0_15px_rgba(107,33,168,0.15)]",
    cyan: "border-neon/30 shadow-[0_0_15px_rgba(0,243,255,0.1)]",
  };

  return (
    <div className={clsx(
      "relative bg-obsidian/80 backdrop-blur-sm border p-6",
      glowStyles[glow],
      className
    )}>
      {/* Corner Accents */}
      <div className="absolute top-0 left-0 w-2 h-2 border-t border-l border-current opacity-70" />
      <div className="absolute top-0 right-0 w-2 h-2 border-t border-r border-current opacity-70" />
      <div className="absolute bottom-0 left-0 w-2 h-2 border-b border-l border-current opacity-70" />
      <div className="absolute bottom-0 right-0 w-2 h-2 border-b border-r border-current opacity-70" />

      {title && (
        <div className="mb-4 flex items-center gap-2 border-b border-white/10 pb-2">
          <div className={`w-2 h-2 rounded-full ${glow === 'red' ? 'bg-blood' : glow === 'purple' ? 'bg-void' : 'bg-neon'} animate-pulse`} />
          <h3 className="font-display text-sm tracking-widest uppercase text-gray-300">{title}</h3>
        </div>
      )}
      
      {children}
    </div>
  );
};
