import React from 'react';
import TrainingSessionPage from './TrainingSessionPage';
import GameDetailPage from './GameDetailPage';
import { WebSocketProvider } from './WebSocketContext';
import { BrowserRouter as BrowserRouter, Routes, Route } from 'react-router-dom';
import NavBar from './NavBar';

function App() {
    return (
        <div className="App">
            <WebSocketProvider>
                <BrowserRouter>
                    <NavBar />
                    <Routes>
                        <Route path="/" element={<TrainingSessionPage />} />
                        <Route path="/training" element={<TrainingSessionPage />} />
                        <Route path="/game_detail/:uuid" element={<GameDetailPage /> } />
                    </Routes>
                </BrowserRouter>
            </WebSocketProvider>
        </div>
    );
}

export default App;
