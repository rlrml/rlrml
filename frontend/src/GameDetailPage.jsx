import { useParams } from 'react-router-dom'

const GameDetailPage = () => {
	const { uuid } = useParams()
    return (
        <div>
			{ uuid }
        </div>
    );
}

export default GameDetailPage;
