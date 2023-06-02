import { useParams } from 'react-router-dom'

const TrainingSessionPage = () => {
	const { uuid } = useParams()
    return (
        <div>
			{ uuid }
        </div>
    );
}

export default TrainingSessionPage;
