import React from 'react'
import {Row, Card} from "react-bootstrap";
import { withRouter } from 'react-router-dom'

const LandingPage = (props) => (
  <div>
    <Row>
      <a style={{ cursor: 'pointer' }} onClick={() => props.history.push('/me/home')}>
        <Card className="card-item">
          <Card.Body>
            <Card.Title>Demetres side project</Card.Title>
            <Card.Text className="text-muted">Movie review sentiment analysis</Card.Text>
          </Card.Body>
        </Card>
      </a>
    </Row>
  </div>
);


export default withRouter(LandingPage)


