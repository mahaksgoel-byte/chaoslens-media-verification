import { Panel } from '../components/ui/Panel';
import { GlitchButton } from '../components/ui/GlitchButton';
import { ToggleLeft, ToggleRight, Settings as SettingsIcon, Save, Trash2, AlertTriangle } from 'lucide-react';
import { useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { deleteDoc } from 'firebase/firestore';
import { deleteUser } from 'firebase/auth';
import { useNavigate } from 'react-router-dom';
import {
  EmailAuthProvider,
  reauthenticateWithCredential,
} from 'firebase/auth';
import { auth, db } from '../firebase';
import { doc, getDoc, updateDoc, serverTimestamp } from 'firebase/firestore';

const SettingRow = ({ label, description, active, onToggle }: any) => (
  <motion.div
    whileHover={{ x: 4 }}
    className="flex items-center justify-between p-5 border-b border-white/5 last:border-0 group cursor-pointer"
    onClick={onToggle}
  >
    <div className="flex-1">
      <h4 className="font-semibold text-white mb-1" style={{ fontFamily: '"Playfair Display", Georgia, serif', fontSize: '1.1rem' }}>
        {label}
      </h4>
      <p className="text-xs font-mono text-gray-500">{description}</p>
    </div>
    
    <motion.button
      whileHover={{ scale: 1.1 }}
      whileTap={{ scale: 0.9 }}
      className={`transition-colors ml-6 ${active ? 'text-cyan-400' : 'text-gray-600'}`}
    >
      {active ? <ToggleRight className="w-10 h-10" /> : <ToggleLeft className="w-10 h-10" />}
    </motion.button>
  </motion.div>
);

export const Settings = () => {
  const user = auth.currentUser;
  const navigate = useNavigate();
  const [showRift, setShowRift] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [passwordConfirm, setPasswordConfirm] = useState('');
  const [authError, setAuthError] = useState<string | null>(null);

  const [settings, setSettings] = useState({
    sound: true,
    notifications: true
  });

  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);

  // Load settings from Firestore
  useEffect(() => {
    if (!user) return;

    const loadSettings = async () => {
      const ref = doc(db, 'settings', user.uid);
      const snap = await getDoc(ref);

      if (snap.exists()) {
        const data = snap.data();
        setSettings({
          sound: data.audio,
          notifications: data.notifications
        });
      }

      setLoading(false);
    };

    loadSettings();
  }, [user]);

  const toggle = (key: keyof typeof settings) => {
    setSettings(prev => ({ ...prev, [key]: !prev[key] }));
  };

  const confirmDeleteAccount = async () => {
    if (!user || !user.email) return;

    try {
      setDeleting(true);
      setAuthError(null);

      const credential = EmailAuthProvider.credential(
        user.email,
        passwordConfirm
      );

      await reauthenticateWithCredential(user, credential);
      await deleteDoc(doc(db, 'settings', user.uid));
      await deleteDoc(doc(db, 'users', user.uid));
      await deleteUser(user);

      setShowRift(false);
      navigate('/signup');
    } catch (err: any) {
      console.error('Account deletion failed:', err);

      if (err.code === 'auth/wrong-password') {
        setAuthError('Incorrect password.');
      } else if (err.code === 'auth/requires-recent-login') {
        setAuthError('Please re-authenticate.');
      } else {
        setAuthError('Deletion failed.');
      }

      setDeleting(false);
    }
  };

  const saveSettings = async () => {
    if (!user) return;

    setSaving(true);
    const ref = doc(db, 'settings', user.uid);

    await updateDoc(ref, {
      audio: settings.sound,
      notifications: settings.notifications,
      updatedAt: serverTimestamp()
    });

    setTimeout(() => setSaving(false), 1000);
  };

  if (loading) return null;

  return (
    <div className="h-full flex flex-col gap-8">
      {/* HEADER */}
      <motion.header
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
        className="flex items-center gap-6"
      >
        <motion.div
          animate={{ rotate: [0, 5, -5, 0] }}
          transition={{ duration: 3, repeat: Infinity }}
          className="w-16 h-16 rounded-2xl bg-cyan-950/30 border border-cyan-500/30 flex items-center justify-center"
        >
          <SettingsIcon className="w-8 h-8 text-cyan-400" />
        </motion.div>

        <div>
          <h2
            className="text-5xl font-bold mb-2"
            style={{
              fontFamily: '"Playfair Display", Georgia, serif',
              background: 'linear-gradient(135deg, #ffffff 0%, #00ffff 100%)',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
            }}
          >
            System Configuration
          </h2>
          <p className="text-gray-500 font-mono text-sm tracking-[0.15em] uppercase">
            Customize Your Interface Parameters
          </p>
        </div>
      </motion.header>

      {/* INTERFACE SETTINGS */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
      >
        <Panel title="Interface Parameters" glow="cyan">
          <SettingRow
            label="Audio Feedback"
            description="Enable UI interaction sounds and system alerts"
            active={settings.sound}
            onToggle={() => toggle('sound')}
          />
          <SettingRow
            label="Notifications"
            description="Allow system alerts, updates, and real-time threat notifications"
            active={settings.notifications}
            onToggle={() => toggle('notifications')}
          />
        </Panel>
      </motion.div>

      {/* SAVE BUTTON */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3 }}
        className="flex justify-end"
      >
        <motion.button
          onClick={saveSettings}
          disabled={saving}
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          className="group relative px-8 py-4 bg-gradient-to-r from-cyan-600 to-cyan-700 hover:from-cyan-500 hover:to-cyan-600 text-white font-mono text-sm tracking-[0.15em] uppercase font-semibold rounded-xl overflow-hidden disabled:opacity-50"
        >
          <motion.div
            className="absolute inset-0 bg-gradient-to-r from-white/0 via-white/20 to-white/0"
            animate={{ x: ['-100%', '200%'] }}
            transition={{ duration: 2, repeat: Infinity }}
          />
          <span className="relative z-10 flex items-center gap-2">
            {saving ? (
              <>
                <motion.div
                  animate={{ rotate: 360 }}
                  transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                >
                  <Save className="w-4 h-4" />
                </motion.div>
                Saving...
              </>
            ) : (
              <>
                <Save className="w-4 h-4" />
                Save Settings
              </>
            )}
          </span>
        </motion.button>
      </motion.div>

      {/* DANGER ZONE */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4 }}
      >
        <Panel title="Danger Zone" glow="red" className="border-red-500/30">
          <div className="flex items-center justify-between p-6">
            <div className="flex-1">
              <h4
                className="font-bold text-red-500 text-xl mb-2"
                style={{ fontFamily: '"Playfair Display", Georgia, serif' }}
              >
                Delete Account Permanently
              </h4>
              <p className="text-sm font-mono text-gray-500">
                This will permanently delete your account and all associated data.
                <br />
                <span className="text-red-400/80">This action cannot be undone.</span>
              </p>
            </div>

            <motion.button
              onClick={() => setShowRift(true)}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              className="group relative ml-6 px-6 py-3 bg-gradient-to-r from-red-600 to-red-700 hover:from-red-500 hover:to-red-600 text-white font-mono text-xs tracking-[0.15em] uppercase font-semibold rounded-xl overflow-hidden"
            >
              <motion.div
                className="absolute inset-0 bg-gradient-to-r from-white/0 via-white/20 to-white/0"
                animate={{ x: ['-100%', '200%'] }}
                transition={{ duration: 2, repeat: Infinity }}
              />
              <span className="relative z-10 flex items-center gap-2">
                <Trash2 className="w-4 h-4" />
                Delete Account
              </span>
            </motion.button>
          </div>
        </Panel>
      </motion.div>

      {/* DELETION CONFIRMATION MODAL */}
      <AnimatePresence>
        {showRift && (
          <>
            {/* Split screen effect */}
            <motion.div
              className="fixed inset-y-0 left-0 w-1/2 z-40 bg-gradient-to-r from-black to-transparent"
              initial={{ x: 0 }}
              animate={{ x: '-100%' }}
              exit={{ x: 0 }}
              transition={{ duration: 0.6, ease: 'easeInOut' }}
            />
            <motion.div
              className="fixed inset-y-0 right-0 w-1/2 z-40 bg-gradient-to-l from-black to-transparent"
              initial={{ x: 0 }}
              animate={{ x: '100%' }}
              exit={{ x: 0 }}
              transition={{ duration: 0.6, ease: 'easeInOut' }}
            />

            {/* Backdrop */}
            <motion.div
              className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-md"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={() => setShowRift(false)}
            >
              {/* Modal */}
              <motion.div
                initial={{ scale: 0.8, y: 50 }}
                animate={{ scale: 1, y: 0 }}
                exit={{ scale: 0.8, y: 50 }}
                onClick={(e) => e.stopPropagation()}
                className="relative w-full max-w-lg mx-4 bg-gradient-to-br from-black via-red-950/20 to-black border border-red-500/30 rounded-2xl overflow-hidden"
                style={{
                  boxShadow: '0 0 80px rgba(255,0,0,0.4), inset 0 0 60px rgba(255,0,0,0.05)',
                }}
              >
                {/* Animated background */}
                <motion.div
                  className="absolute inset-0 opacity-30"
                  animate={{
                    background: [
                      'radial-gradient(circle at 50% 50%, rgba(255,0,0,0.3), transparent 70%)',
                      'radial-gradient(circle at 30% 70%, rgba(255,0,0,0.3), transparent 70%)',
                      'radial-gradient(circle at 70% 30%, rgba(255,0,0,0.3), transparent 70%)',
                      'radial-gradient(circle at 50% 50%, rgba(255,0,0,0.3), transparent 70%)',
                    ],
                  }}
                  transition={{ duration: 6, repeat: Infinity }}
                />

                {/* Header */}
                <div className="relative z-10 px-8 py-6 border-b border-red-500/30 text-center">
                  <motion.div
                    animate={{ scale: [1, 1.1, 1] }}
                    transition={{ duration: 2, repeat: Infinity }}
                    className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-red-950/50 border-2 border-red-500 mb-4"
                  >
                    <AlertTriangle className="w-8 h-8 text-red-500" />
                  </motion.div>

                  <h3
                    className="text-3xl font-bold text-red-500 mb-2"
                    style={{ fontFamily: '"Playfair Display", Georgia, serif' }}
                  >
                    Account Deletion Confirmation
                  </h3>
                  <p className="text-sm font-mono text-gray-400">
                    Authorization Required
                  </p>
                </div>

                {/* Content */}
                <div className="relative z-10 px-8 py-8 space-y-6">
                  <p className="text-sm font-mono text-gray-300 text-center leading-relaxed">
                    This action will permanently delete your account and all associated data.
                    <br />
                    <span className="text-red-400">All data will be irreversibly destroyed.</span>
                  </p>

                  {/* Password input */}
                  <div className="space-y-2">
                    <label className="block text-xs font-mono text-gray-500 tracking-wide uppercase">
                      Confirm Password
                    </label>
                    <input
                      type="password"
                      placeholder="Enter your password"
                      value={passwordConfirm}
                      onChange={(e) => setPasswordConfirm(e.target.value)}
                      className="w-full bg-black/60 border border-red-500/30 focus:border-red-500 text-white px-4 py-3 font-mono text-sm outline-none rounded-xl transition-all"
                    />
                  </div>

                  {/* Error message */}
                  {authError && (
                    <motion.div
                      initial={{ opacity: 0, y: -10 }}
                      animate={{ opacity: 1, y: 0 }}
                      className="flex items-center gap-2 p-3 bg-red-950/30 border border-red-500/30 rounded-lg"
                    >
                      <AlertTriangle className="w-4 h-4 text-red-400 flex-shrink-0" />
                      <p className="text-xs text-red-400 font-mono">
                        {authError}
                      </p>
                    </motion.div>
                  )}

                  {/* Actions */}
                  <div className="flex gap-3 pt-4">
                    <motion.button
                      onClick={() => setShowRift(false)}
                      disabled={deleting}
                      whileHover={{ scale: 1.02 }}
                      whileTap={{ scale: 0.98 }}
                      className="flex-1 px-4 py-3 bg-black/60 border border-white/10 hover:border-white/30 text-white rounded-xl font-mono text-sm tracking-wide transition-all"
                    >
                      Abort
                    </motion.button>

                    <motion.button
                      onClick={confirmDeleteAccount}
                      disabled={deleting || !passwordConfirm}
                      whileHover={{ scale: 1.02 }}
                      whileTap={{ scale: 0.98 }}
                      className="flex-1 px-4 py-3 bg-gradient-to-r from-red-600 to-red-700 hover:from-red-500 hover:to-red-600 text-white rounded-xl font-mono text-sm tracking-wide transition-all disabled:opacity-50 disabled:cursor-not-allowed relative overflow-hidden group"
                    >
                      <motion.div
                        className="absolute inset-0 bg-gradient-to-r from-white/0 via-white/20 to-white/0"
                        animate={{ x: ['-100%', '200%'] }}
                        transition={{ duration: 2, repeat: Infinity }}
                      />
                      <span className="relative z-10">
                        {deleting ? 'Erasing...' : 'Confirm Delete'}
                      </span>
                    </motion.button>
                  </div>
                </div>
              </motion.div>
            </motion.div>
          </>
        )}
      </AnimatePresence>
    </div>
  );
};