import React from 'react'
import {Row, Card} from "react-bootstrap";
import { withRouter } from 'react-router-dom'

const LandingPage = (props) => (
  <div>
    <Row>
      <a style={{ cursor: 'pointer' }} onClick={() => props.history.push('/me/home')}>
        <Card className="card-item">
          <Card.Body>
            <Card.Title>Personal Website</Card.Title>
            <Card.Text className="text-muted">A website about me</Card.Text>
          </Card.Body>
        </Card>
      </a>
    </Row>
    <Row>
      <p>
        Minecraft Server Status Checker
      </p>
    </Row>
  </div>
)


export default withRouter(LandingPage)


