import React, { createContext, useContext, useEffect, useRef, useState } from 'react';
import { auth, db } from '../firebase';
import { doc, onSnapshot } from 'firebase/firestore';
import { onAuthStateChanged, User } from 'firebase/auth';

interface AudioContextType {
  audioEnabled: boolean;
  playClick: () => void;
  playRift: () => void;
  setAudioEnabled: (val: boolean) => void;
}

const AudioContext = createContext<AudioContextType | undefined>(undefined);

export const AudioProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [audioEnabled, setAudioEnabled] = useState(true);
  const [user, setUser] = useState<User | null>(null);

  const clickAudioRef = useRef(new Audio('/click.mp3'));
  const riftAudioRef = useRef(new Audio('/rift-open.mp3'));

  // Track currently logged-in user
  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, (currentUser) => {
      setUser(currentUser);
    });
    return () => unsubscribe();
  }, []);

  // Load audio setting from Firestore whenever user changes
  useEffect(() => {
    if (!user) return;

    const ref = doc(db, 'settings', user.uid);

    // Real-time listener: updates if user changes setting
    const unsubscribe = onSnapshot(ref, (snap) => {
      if (snap.exists()) {
        const data = snap.data();
        if (typeof data.audio === 'boolean') {
          setAudioEnabled(data.audio);
          console.log('AudioProvider: audioEnabled updated from Firestore:', data.audio);
        }
      }
    }, (err) => {
      console.error('AudioProvider: failed to read settings', err);
    });

    return () => unsubscribe();
  }, [user]);

  // Keep ref updated for global click listener
  const audioEnabledRef = useRef(audioEnabled);
  useEffect(() => {
    audioEnabledRef.current = audioEnabled;
  }, [audioEnabled]);

  const playClick = () => {
    if (!audioEnabledRef.current) return;
    const audio = clickAudioRef.current;
    audio.currentTime = 0;
    audio.play().catch(() => {});
  };

  const playRift = () => {
    if (!audioEnabledRef.current) return;
    const audio = riftAudioRef.current;
    audio.currentTime = 0;
    audio.play().catch(() => {});
  };

  // Global click listener
  useEffect(() => {
    const handleClick = () => {
      if (!audioEnabledRef.current) return;
      const audio = clickAudioRef.current;
      audio.currentTime = 0;
      audio.play().catch(() => {});
    };

    document.addEventListener('click', handleClick);
    return () => document.removeEventListener('click', handleClick);
  }, []);

  return (
    <AudioContext.Provider value={{ audioEnabled, playClick, playRift, setAudioEnabled }}>
      {children}
    </AudioContext.Provider>
  );
};

export const useAudio = () => {
  const context = useContext(AudioContext);
  if (!context) throw new Error('useAudio must be used within AudioProvider');
  return context;
};
