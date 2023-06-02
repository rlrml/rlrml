import React from 'react';
import _ from 'lodash';

const WebSocketContext = React.createContext(null);

const WebSocketProvider = ({ children }) => {
  const [connectionStatus, setConnectionStatus] = React.useState('disconnected');
  const [webSocketAddress, setWebSocketAddress] = React.useState(null);
  const [webSocket, setWebSocket] = React.useState(null);

  const [sorting, setSorting] = React.useState([]);
  const [lossHistory, setLossHistory] = React.useState([]);
  const [gameInfo, setGameInfo] = React.useState({});
  const [trainingPlayerCount, setTrainingPlayerCount] = React.useState(4);

  const [configuration, setConfiguration] = React.useState({});
  let socket;

  const handleLossBatch = (data) => {
    const newData = getGameInfo(data);
    setGameInfo(prevGameInfo => ({...prevGameInfo, ...newData}));
  }

  const handleTrainingEpoch = (data) => {
    setLossHistory(prevLossHistory => [...prevLossHistory, data.loss]);
    const newData = getGameInfo(data);
    setGameInfo(prevGameInfo => ({...prevGameInfo, ...newData}));
  };

  const handleTrainingStart = (data) => {
    if (isNaN(data.player_count)) {
      console.log(`Data did not contain numerical player count ${data}`)
    }
    setTrainingPlayerCount(data.player_count);
  }

  const makeWebsocketRequest = (type, data) => {
    if (webSocket && webSocket.readyState === WebSocket.OPEN) {
      webSocket.send(JSON.stringify({ type, data }));
    }
  }

  const messageTypeToHandler = {
    "training_epoch": handleTrainingEpoch,
    "training_start": handleTrainingStart,
    "loss_batch": handleLossBatch,
  };

  const processGameData = (data, uuid, tracker_suffixes, y, y_pred, masks) => {
    return [uuid, {
      "uuid": uuid,
      "players": _.zipWith(
        tracker_suffixes, y, y_pred, masks,
        (tracker_suffix, mmr, prediction, mask) => {
          return {tracker_suffix, mmr, prediction, mask};
        }
      ),
      y_pred,
      y,
      masks,
      "update_epoch": data.epoch,
    }]
  }

  const getGameInfo = (data) => {
    const zipped = _.zip(data.uuids, data.tracker_suffixes, data.y, data.y_pred, data.mask);
    return Object.fromEntries(zipped.map((args) => processGameData(data, ...args)));
  }

  React.useEffect(() => {
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
      const message = JSON.parse(event.data);
      if (message.type in messageTypeToHandler) {
        messageTypeToHandler[message.type](message.data);
      } else {
        console.log(`Unable to handle message of type ${message.type}`)
      }
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
	<WebSocketContext.Provider value={{ lossHistory, gameInfo, connectionStatus, setWebSocketAddress, webSocket, trainingPlayerCount, makeWebsocketRequest, sorting, setSorting }}>
      {children}
    </WebSocketContext.Provider>
  );
};

export { WebSocketContext, WebSocketProvider };
