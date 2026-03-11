import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { Layout } from "./components/Layout";
import { Login } from "./pages/Login";
import { Signup } from "./pages/Signup";
import { Dashboard } from "./pages/Dashboard";
import { Misinfo } from "./pages/Misinfo";
import { Settings } from "./pages/Settings";
import { AuthGuard } from "./components/AuthGuard";
import { AudioProvider } from './context/AudioContext';
import { useEffect } from "react";
import { HomePage } from './pages/HomePage'; 

import { initExtensionBridge } from "./extensionBridge";

function App() {
  useEffect(() => {
    initExtensionBridge();
  }, []);

  useEffect(() => {
    window.addEventListener("message", (e) => {
      console.log("🌍 Website received message:", e.data);
    });
  }, []);
  
  return (
    <AudioProvider>
    <BrowserRouter>
      <Routes>
        {/* Default */}
        <Route path="/" element={<Navigate to="/login" />} />

        {/* Public */}
        <Route path="/login" element={<Login />} />
        <Route path="/signup" element={<Signup />} />

        {/* Protected */}
        <Route
          path="/home"
          element={
            <AuthGuard>
              <Layout />
            </AuthGuard>
          }
        >
          <Route index element={<HomePage />} />
          <Route path="deepfake" element={<Dashboard/>} />
          <Route path="misinfo" element={<Misinfo />} />
          <Route path="settings" element={<Settings />} />
        </Route>
      </Routes>
    </BrowserRouter>
    </AudioProvider>
  );
}

export default App;