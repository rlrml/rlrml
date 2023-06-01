import React from 'react';

const WebSocketContext = React.createContext(null);

const WebSocketProvider = ({ children }) => {
  const [lossHistory, setLossHistory] = React.useState([]);
  const [gameInfo, setGameInfo] = React.useState({});
  const [connectionStatus, setConnectionStatus] = React.useState('disconnected');
  const [webSocketAddress, setWebSocketAddress] = React.useState(null);
  const [webSocket, setWebSocket] = React.useState(null);
  let socket;

  React.useEffect(() => {
    if (!webSocketAddress) {
      return;
    }

    console.log("Attempting connection to " + webSocketAddress);

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
      setWebSocketAddress(null);
	};

    return () => {
      socket.close();
      setWebSocketAddress(null);
    };

  }, [webSocketAddress]);

  return (
	<WebSocketContext.Provider value={{ lossHistory, gameInfo, connectionStatus, setWebSocketAddress, webSocket }}>
      {children}
    </WebSocketContext.Provider>
  );
};

function getGameInfo(data) {
  return Object.fromEntries(data.uuids.map((uuid, index) => [uuid, {
    "uuid": uuid,
    "y": data.y[index],
    "y_pred": data.y_pred[index],
    "y_loss": data.y_loss[index],
    "update_epoch": data.epoch,
  }]));
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
