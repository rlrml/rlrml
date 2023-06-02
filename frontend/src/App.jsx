import React from 'react';
import TrainingSessionPage from './TrainingSessionPage';
import LossAnalysisPage from './LossAnalysisPage';
import GameDetailPage from './GameDetailPage';
import PlayerDetailPage from './PlayerDetailPage';
import { WebSocketProvider } from './WebSocketContext';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
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
                        <Route path="/loss_analysis" element={<LossAnalysisPage />} />
                        <Route path="/game_detail/:uuid" element={<GameDetailPage /> } />
                        <Route path="/player_detail/:trackerType/:trackerId" element={<PlayerDetailPage /> } />
                    </Routes>
                </BrowserRouter>
            </WebSocketProvider>
        </div>
    );
}

export default App;
