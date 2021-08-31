from flask_cors import cross_origin
from . import routes


# endpoint to help with testing backend is running without relying on the sentinment_model being loaded correctly
@routes.route('/')
@cross_origin()
def get_homepage():
    # TODO: add example
    return """
        <h1 id="documentation">Documentation</h1>
        <h2 id="endpoints">Endpoints</h2>
        <p>All endpoint routes are prefixed with /api/v1</p>
        <ul>
        <li>get_sentiment</li>
        </ul>
        <p>endpoint takes &#39;phrase&#39; as param and returns one of 
        <code>[NEGATIVE, SOMEWHAT_NEGATIVE, NEUTRAL, SOMEWHAT_POSITIVE, POSITIVE]</code></p>
        <ul>
        <br>
        <li>time</li>
        </ul>
        <p>Takes no input, returns current posix timestamp. Used mainly for testing</p>
        <br>
    """
