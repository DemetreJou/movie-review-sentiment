import React from 'react';
import './App.css';
import {BrowserRouter, Route, Switch} from 'react-router-dom';
import MeHome from './me/home.js'
import LandingPage from "./landingPage";

function App() {
  return (
    <div>
      <BrowserRouter>
        <Switch>
          <Route exact path='/' component={LandingPage} />
          <Route exact path='/me/home' component={MeHome}/>
        </Switch>
      </BrowserRouter>
    </div>
  );
}

export default App;
