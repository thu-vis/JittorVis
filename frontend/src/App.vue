<template>
  <div id="app">
    <div id="statistic-container">
      <statistic></statistic>
    </div>
    <div id="tree-container">
      <div id="navi-container"><navigation></navigation></div>
      <div id="network-container"><network></network></div>
    </div>
  </div>
</template>

<script>
import Navigation from './components/Navigation.vue'
import Network from './components/Network.vue'
import Statistic from './components/Statistic.vue'

export default {
  components: { Network, Navigation, Statistic },
  name: 'App'
}
</script>

<style>
html, body, #app {
  margin: 0;
  width: 100%;
  height: 100%;
}

#app {
  font-family: 'Avenir', Helvetica, Arial, sans-serif;
  display: flex;
  flex-direction: column;
}

#statistic-container {
  width: 100%;
  height: 20%;
}

#tree-container {
  width: 100%;
  height: 80%;
  display: flex;
}

#navi-container {
  width: 30%;
  height: 100%;
}

#network-container {
  width: 70%;
  height: 100%;
}

</style>
