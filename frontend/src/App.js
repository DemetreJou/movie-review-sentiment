import React, {useEffect, useState} from 'react';
import './App.css';
import {BrowserRouter, Route, Switch} from 'react-router-dom';
import LandingPage from "./Pages/landingPage"
import myComponent from "./Pages/api_call"

// function App() {
//   return (
//     <div>
//       <BrowserRouter>
//         <Switch>
//           <Route exact path='/' component={LandingPage} />
//             <Route exact path='/test' component={myComponent}/>
//         </Switch>
//       </BrowserRouter>
//     </div>
//   );
// }


function App() {
  const [currentTime, setCurrentTime] = useState(0);

  useEffect(() => {
    fetch('http://localhost:5000/api/v1/time').then(res => res.json()).then(data => {
      setCurrentTime(data.time);
    });
  }, []);

  return (
    <div className="App">
      <header className="App-header">
        <p>The current time is {currentTime}.</p>
      </header>
    </div>
  );
}

export default App;
