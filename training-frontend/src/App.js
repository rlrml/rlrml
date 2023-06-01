import React from 'react';
import TrainingSessionPage from './TrainingSessionPage';
import { WebSocketProvider } from './WebSocketContext';

function App() {
    return (
            <div className="App">
              <WebSocketProvider>
                 <TrainingSessionPage />
              </WebSocketProvider>
            </div>
    );
}

export default App;
