import React from 'react'
import { withRouter } from 'react-router-dom'

class myComponent extends React.Component {
  componentDidMount() {
    const apiUrl = 'localhost:5000/test';
    fetch(apiUrl)
      // .then((response) => response.json())
      .then((data) => {
        console.log('This is your data', data);
      })
  }
  render() {
    return <h1>check console</h1>;
  }
}
export default withRouter(myComponent);



