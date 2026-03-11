import { useState } from "react";
import { Outlet, Link, useLocation, useNavigate } from "react-router-dom";
import { CustomCursor } from "./CustomCursor";
import {
  ShieldAlert,
  Activity,
  Settings,
  LogOut,
  Home,
  ChevronRight,
} from "lucide-react";
import { clsx } from "clsx";
import { motion } from "framer-motion";
import MadMaxChatOrb from "./MadMaxChatOrb";
import { signOut } from "firebase/auth";
import { auth } from "../firebase";

/* =========================
   Sidebar Item with Hover Effects
========================= */
const SidebarItem = ({
  to,
  icon: Icon,
  label,
  active,
  onClick,
  collapsed,
}: {
  to?: string;
  icon: any;
  label: string;
  active: boolean;
  onClick?: () => void;
  collapsed?: boolean;
}) => {
  const [isHovered, setIsHovered] = useState(false);

  const content = (
    <motion.div
      className={clsx(
        "relative flex items-center gap-4 font-mono transition-all cursor-pointer group overflow-hidden",
        collapsed ? "px-3 py-4 justify-center" : "px-6 py-4",
        active
          ? "text-red-500"
          : "text-gray-500 hover:text-white"
      )}
      onClick={onClick}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      whileHover={{ x: 6 }}
      transition={{ duration: 0.2 }}
    >
      {/* Animated background on hover */}
      <motion.div
        className="absolute inset-0 bg-gradient-to-r from-red-500/0 via-red-500/5 to-red-500/0"
        initial={{ x: "-100%" }}
        animate={{ x: isHovered ? "100%" : "-100%" }}
        transition={{ duration: 0.6, ease: "easeInOut" }}
      />

      {/* Active indicator */}
      {active && (
        <>
          <motion.div
            layoutId="active-pill"
            className="absolute inset-0 bg-gradient-to-r from-red-950/40 to-transparent border-l-2 border-red-500"
            transition={{ type: "spring", stiffness: 300, damping: 30 }}
          />
          <motion.div
            className="absolute left-0 inset-y-0 w-0.5 bg-red-500 shadow-[0_0_20px_rgba(255,0,51,0.8)]"
            layoutId="active-glow"
          />
        </>
      )}

      {/* Icon with rotation animation */}
      <motion.div
        animate={{ rotate: active ? [0, 5, -5, 0] : 0 }}
        transition={{ duration: 0.5 }}
        className="relative z-10"
      >
        <Icon className={clsx("w-5 h-5", active && "drop-shadow-[0_0_8px_rgba(255,0,51,0.8)]")} />
      </motion.div>

      {/* Label */}
      {!collapsed && (
        <span className="relative z-10 font-semibold text-sm tracking-tight">{label}</span>
      )}

      {/* Hover pulse effect */}
      {isHovered && !active && (
        <motion.div
          className="absolute right-0 w-1 h-full bg-gradient-to-b from-transparent via-red-500/50 to-transparent"
          initial={{ opacity: 0 }}
          animate={{ opacity: [0, 1, 0] }}
          transition={{ duration: 1, repeat: Infinity }}
        />
      )}
    </motion.div>
  );

  return to ? (
    <Link to={to}>
      {content}
    </Link>
  ) : (
    <div>{content}</div>
  );
};

/* =========================
   Main Layout
========================= */
export const Layout = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const isAuthPage = ["/login", "/signup"].includes(location.pathname);

  const handleLogout = async () => {
    await signOut(auth);
    navigate("/login");
  };

  return (
    <div className="min-h-screen bg-black text-gray-200 font-mono overflow-hidden">
      <CustomCursor />

      {/* Scanline effect */}
      <div className="pointer-events-none fixed inset-0 z-50 mix-blend-overlay opacity-[0.03] bg-[linear-gradient(rgba(255,255,255,0.03)_1px,transparent_1px)] bg-[size:100%_2px] animate-[scan_8s_linear_infinite]" />
      
      {/* Vignette */}
      <div className="pointer-events-none fixed inset-0 z-40 bg-[radial-gradient(ellipse_at_center,transparent_0%,rgba(0,0,0,0.4)_100%)]" />

      {isAuthPage ? (
        <Outlet />
      ) : (
        <div className="flex h-screen relative">
          {/* Ambient background glow */}
          <div className="pointer-events-none fixed inset-0 z-0">
            <motion.div
              className="absolute top-0 left-0 w-[800px] h-[800px] rounded-full blur-[180px]"
              style={{
                background: 'radial-gradient(circle, rgba(255,0,51,0.15) 0%, transparent 70%)',
              }}
              animate={{
                scale: [1, 1.2, 1],
                opacity: [0.3, 0.5, 0.3],
              }}
              transition={{ duration: 8, repeat: Infinity, ease: "easeInOut" }}
            />
            <motion.div
              className="absolute bottom-0 right-0 w-[600px] h-[600px] rounded-full blur-[160px]"
              style={{
                background: 'radial-gradient(circle, rgba(0,255,255,0.1) 0%, transparent 70%)',
              }}
              animate={{
                scale: [1.2, 1, 1.2],
                opacity: [0.2, 0.4, 0.2],
              }}
              transition={{ duration: 10, repeat: Infinity, ease: "easeInOut", delay: 1 }}
            />
          </div>

          {/* =========================
              SIDEBAR
          ========================= */}
          <motion.aside
            initial={{ x: -300, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            transition={{ duration: 0.6, ease: "easeOut" }}
            className={clsx(
              "hidden md:flex flex-col bg-black/40 backdrop-blur-xl border-r border-white/5 z-30 relative transition-all duration-300",
              sidebarCollapsed ? "w-20" : "w-80"
            )}
            style={{
              background: 'linear-gradient(180deg, rgba(0,0,0,0.95) 0%, rgba(10,0,0,0.98) 100%)',
            }}
          >
            {/* Toggle button - Vertically centered on whole sidebar, at right edge */}
            <motion.button
              onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
              className="absolute right-2 top-1/2 -translate-y-1/2 flex items-center justify-center p-2 text-gray-500 hover:text-red-400 transition-colors border border-white/5 hover:border-red-500/30 rounded-lg bg-black/20 hover:bg-red-950/20 z-40"
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              title={sidebarCollapsed ? "Expand Sidebar" : "Collapse Sidebar"}
            >
              <motion.div
                animate={{ rotate: sidebarCollapsed ? 0 : 180 }}
                transition={{ duration: 0.3 }}
              >
                <ChevronRight className="w-4 h-4" />
              </motion.div>
            </motion.button>
            {/* Glowing border effect */}
            <div className="absolute inset-y-0 right-0 w-px bg-gradient-to-b from-transparent via-red-500/30 to-transparent" />

            {/* Header */}
            <div className={clsx(
              "border-b border-white/5 relative transition-all duration-300",
              sidebarCollapsed ? "px-4 py-4" : "px-6 py-6"
            )}>
              {/* Animated circuits background */}
              <div className="absolute inset-0 opacity-10">
                {[...Array(3)].map((_, i) => (
                  <motion.div
                    key={i}
                    className="absolute h-px bg-gradient-to-r from-transparent via-red-500 to-transparent"
                    style={{ top: `${30 + i * 25}%`, left: 0, right: 0 }}
                    animate={{ opacity: [0.2, 0.6, 0.2] }}
                    transition={{ duration: 3, repeat: Infinity, delay: i * 0.5 }}
                  />
                ))}
              </div>

              <motion.div
                initial={{ scale: 0.8, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                transition={{ delay: 0.3, duration: 0.5 }}
                className="relative"
              >
                {/* Logo and Title - No Shield */}
                <div className="flex items-center gap-3 mb-4">
                  <div className={sidebarCollapsed ? "hidden" : ""}>
                    <h1
                      className="text-3xl font-bold tracking-[0.3em] leading-none"
                      style={{
                        fontFamily: '"Playfair Display", Georgia, serif',
                        background: 'linear-gradient(135deg, #ffffff 0%, #ff0033 100%)',
                        WebkitBackgroundClip: 'text',
                        WebkitTextFillColor: 'transparent',
                      }}
                    >
                      ChaosLens
                    </h1>
                  </div>
                </div>

                {/* Subtitle */}
                {!sidebarCollapsed && (
                  <div className="space-y-1">
                    <p className="text-[9px] text-gray-600 tracking-[0.25em] uppercase font-bold">
                      Deepfake Detection System
                    </p>
                    <div className="flex items-center gap-2">
                      <motion.div
                        className="w-1.5 h-1.5 rounded-full bg-red-500"
                        animate={{ opacity: [0.3, 1, 0.3], scale: [1, 1.2, 1] }}
                        transition={{ duration: 2, repeat: Infinity }}
                      />
                      <p className="text-[8px] text-red-500/80 tracking-[0.2em] uppercase font-semibold">
                        Analysis Interface
                      </p>
                    </div>
                  </div>
                )}
              </motion.div>
            </div>

            {/* Navigation */}
            <nav className="flex-1 py-4 space-y-1 px-4">
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.4, staggerChildren: 0.1 }}
                className="space-y-1"
              >
                <SidebarItem
                  to="/home"
                  icon={Home}
                  label="Dashboard"
                  active={location.pathname === "/home"}
                  collapsed={sidebarCollapsed}
                />

                <SidebarItem
                  to="/home/deepfake"
                  icon={Activity}
                  label="Detection"
                  active={location.pathname === "/home/deepfake"}
                  collapsed={sidebarCollapsed}
                />

                <SidebarItem
                  to="/home/misinfo"
                  icon={ShieldAlert}
                  label="Analysis"
                  active={location.pathname === "/home/misinfo"}
                  collapsed={sidebarCollapsed}
                />

                <SidebarItem
                  to="/home/settings"
                  icon={Settings}
                  label="Settings"
                  active={location.pathname === "/home/settings"}
                  collapsed={sidebarCollapsed}
                />
              </motion.div>

              {/* Decorative divider */}
              <div className="py-6 px-2">
                <div className="h-px bg-gradient-to-r from-transparent via-white/10 to-transparent" />
              </div>
            </nav>

            {/* Footer */}
            <div className={clsx(
              "border-t border-white/5 space-y-4 transition-all duration-300",
              sidebarCollapsed ? "px-2 py-4" : "px-6 py-6"
            )}>
              {/* Logout Button */}
              <motion.button
                onClick={handleLogout}
                className={clsx(
                  "flex items-center gap-3 text-gray-400 hover:text-red-400 transition-colors text-xs tracking-[0.15em] uppercase font-semibold border border-white/5 hover:border-red-500/30 rounded-lg bg-black/20 hover:bg-red-950/20 group relative overflow-hidden",
                  sidebarCollapsed ? "w-auto px-3 py-3 justify-center" : "w-full px-4 py-3"
                )}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                title={sidebarCollapsed ? "Logout" : "Log Out"}
              >
                <motion.div
                  className="absolute inset-0 bg-gradient-to-r from-red-500/0 via-red-500/10 to-red-500/0"
                  initial={{ x: "-100%" }}
                  whileHover={{ x: "100%" }}
                  transition={{ duration: 0.6 }}
                />
                <LogOut className="w-4 h-4 relative z-10" />
                {!sidebarCollapsed && (
                  <span className="relative z-10">Log Out</span>
                )}
              </motion.button>

              {/* Footer text */}
              {!sidebarCollapsed && (
                <div className="text-center">
                  <p className="text-[8px] text-gray-500 tracking-[0.2em] uppercase">
                    ChaosLens Protocol
                  </p>
                </div>
              )}
            </div>
          </motion.aside>

          {/* =========================
              MAIN CONTENT
          ========================= */}
          <main className="flex-1 relative z-10 overflow-y-auto">
            <div className="min-h-full p-8 md:p-12">
              <Outlet />
            </div>
          </main>

          {/* =========================
              GLOBAL CHATBOT
          ========================= */}
          <MadMaxChatOrb />
        </div>
      )}

      <style>{`
        @keyframes scan {
          0% { transform: translateY(0); }
          100% { transform: translateY(100%); }
        }
      `}</style>
    </div>
  );
};