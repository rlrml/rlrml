import React, { useState, useEffect } from 'react';

const WebSocketContext = React.createContext(null);

const WebSocketProvider = ({ children }) => {
  const [lossHistory, setLossHistory] = useState([]);
  const [gameInfo, setGameInfo] = useState({});
  const [connectionStatus, setConnectionStatus] = useState('disconnected');
  const [webSocketAddress, setWebSocketAddress] = useState(null);
  const [webSocket, setWebSocket] = useState(null);
  let socket;

  useEffect(() => {
    if (!webSocketAddress) {
      return;
    }

    socket = new WebSocket(webSocketAddress)
    setWebSocket(socket);

    socket.onopen = () => {
      setConnectionStatus('connected');
    };

    socket.onerror = () => {
      setConnectionStatus('error');
    };

    socket.onmessage = (event) => {
      const data = JSON.parse(event.data);
      setLossHistory(prevLossHistory => [...prevLossHistory, data.loss]);
      setGameInfo(prevGameInfo => ({...prevGameInfo, ...getGameInfo(data)}));
    };

	socket.onclose = () => {
	  setConnectionStatus('disconnected');
	};

    return () => {
      socket.close();
    };

  }, [webSocketAddress]);

  return (
	<WebSocketContext.Provider value={{ lossHistory, gameInfo, connectionStatus, setWebSocketAddress, webSocket }}>
      {children}
    </WebSocketContext.Provider>
  );
};

function getGameInfo(data) {
	return data.game_info
}

function getLossStats(arr1, arr2) {
  // Calculate MSE
  const squaredDifferences = arr1.map((value, index) => Math.pow(value - arr2[index], 2));
  const mse = squaredDifferences.reduce((sum, value) => sum + value, 0) / arr1.length;

  // Find element with the largest difference
  let maxDiff = 0;
  let maxDiffIndex = -1;
  for (let i = 0; i < arr1.length; i++) {
    const diff = Math.abs(arr1[i] - arr2[i]);
    if (diff > maxDiff) {
      maxDiff = diff;
      maxDiffIndex = i;
    }
  }

  return { mse, maxDiff, maxDiffIndex };
}

function formatArrayWithIndex(array, separator = " | ") {
  const formattedValues = array.map((value, index) => {
    const truncatedValue = Math.trunc(value);
    return `${index + 1}: ${truncatedValue}`;
  });

  return formattedValues.join(separator);
}

export { WebSocketContext, WebSocketProvider };
