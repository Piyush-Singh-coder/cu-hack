import { Route, Routes } from "react-router-dom";
import Main from "./components/Main";
import LandingPage from "./pages/LandingPage";
import { useThemeStore } from "./store/useThemeStore";

// --- Constants ---
export const API_BASE_URL = "https://cu-hack.onrender.com";

export default function App() {

    const {theme} = useThemeStore();
    return (
        <div data-theme = {theme}>
            <Routes>
              <Route path='/' element={<LandingPage/>} />
            <Route path='/ai' element={<Main />}></Route>  
            </Routes>
            
        </div>
    );
}
