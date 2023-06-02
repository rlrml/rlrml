import { useParams } from 'react-router-dom'

const PlayerDetailPage = () => {
	const { trackerType, trackerId } = useParams()
	const trackerSuffix = `${trackerType}/${trackerId}`
    return (
        <div>
			{ trackerType }: { trackerId }
			<br />
			{trackerSuffix}
        </div>
    );
}

export default PlayerDetailPage;
