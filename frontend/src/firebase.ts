// Import the functions you need from the SDKs
import { initializeApp } from "firebase/app";
import { getAuth } from "firebase/auth";
import { getFirestore } from "firebase/firestore";

// Your web app's Firebase configuration
const firebaseConfig = {
  apiKey: "AIzaSyCq3LlVwTn9-iuMuHzqZkMYb9qgHEAflJU",
  authDomain: "madmax-67ca2.firebaseapp.com",
  databaseURL: "https://madmax-67ca2-default-rtdb.firebaseio.com",
  projectId: "madmax-67ca2",
  storageBucket: "madmax-67ca2.firebasestorage.app",
  messagingSenderId: "273549533036",
  appId: "1:273549533036:web:08ff16779252b6bd299d49",
  measurementId: "G-9JGZ6QN9G2"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);

// Export services
export const auth = getAuth(app);
export const db = getFirestore(app);