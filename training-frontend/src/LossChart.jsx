import Chart from 'chart.js/auto';
import {CategoryScale} from 'chart.js';

import React from 'react';
import { Line } from 'react-chartjs-2';
import Slider from 'rc-slider';
import 'rc-slider/assets/index.css';
import { WebSocketContext } from './WebSocketContext';

const LossChart = () => {
    const { lossHistory } = React.useContext(WebSocketContext);

	// Controls the resolution of the graph
    const [bucketSize, setBucketSize] = React.useState(10);

	// A function to calculate bucketed average
    const calculateBucketedAverage = (data, bucketSize) => {
        let bucketedData = [];

        for (let i = 0; i < data.length; i += bucketSize) {
            let bucket = data.slice(i, i + bucketSize);
            let sum = bucket.reduce((a, b) => a + b, 0);
            let avg = sum / bucket.length;
            bucketedData.push(avg);
        }

        return bucketedData;
    };

	// State variable for bucketedLoss
    const [bucketedLossHistory, setBucketedLossHistory] = React.useState([]);

	React.useEffect(() => {
        const bucketedData = calculateBucketedAverage(lossHistory, bucketSize);
        setBucketedLossHistory(bucketedData);

    }, [lossHistory, bucketSize]);

    const options = {
        scales: {
            yAxes: [
                {
                    ticks: {
                        beginAtZero: true,
                    },
                },
            ],
        },
    };

    const handleBucketSizeChange = event => {
        let newSize = Number(event.target.value);
        if(newSize > 0) {
            setBucketSize(newSize);
        }
    };

    return (
        <div>
            Bucket Size:
            <input type="text" onChange={handleBucketSizeChange}
                   placeholder="Enter Bucket Size" />
            <div style={{ margin: '50px' }}>
                <Line
                    data={{
                        labels: bucketedLossHistory.map((_, i) => i + 1),
                        datasets: [
                            {
                                label: 'Training Loss',
                                data: bucketedLossHistory,
                                fill: false,
                                backgroundColor: 'rgb(75, 192, 192)',
                                borderColor: 'rgba(75, 192, 192, 0.2)',
                            },
                        ],
                    }}
                    options={options}
                />
            </div>
        </div>
    );
};

export default LossChart;
