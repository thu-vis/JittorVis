<template>
    <div class="waiting-div">
        <img class="rotating" src="/static/images/refresh.png">
    </div>
</template>

<script>

export default {
    name: 'waiting-icon',
};
</script>

<style>
.waiting-div {
    background-color: #D3D3D3;
    opacity: 0.5;
    position: absolute;
    bottom: 0;
    right: 0;
    top: 0;
    left: 0;
    display: flex;
    justify-content: center;
    align-items: center;
}

@-webkit-keyframes rotating /* Safari and Chrome */ {
  from {
    -webkit-transform: rotate(0deg);
    -o-transform: rotate(0deg);
    transform: rotate(0deg);
  }
  to {
    -webkit-transform: rotate(360deg);
    -o-transform: rotate(360deg);
    transform: rotate(360deg);
  }
}
@keyframes rotating {
  from {
    -ms-transform: rotate(0deg);
    -moz-transform: rotate(0deg);
    -webkit-transform: rotate(0deg);
    -o-transform: rotate(0deg);
    transform: rotate(0deg);
  }
  to {
    -ms-transform: rotate(360deg);
    -moz-transform: rotate(360deg);
    -webkit-transform: rotate(360deg);
    -o-transform: rotate(360deg);
    transform: rotate(360deg);
  }
}
.rotating {
  -webkit-animation: rotating 2s linear infinite;
  -moz-animation: rotating 2s linear infinite;
  -ms-animation: rotating 2s linear infinite;
  -o-animation: rotating 2s linear infinite;
  animation: rotating 2s linear infinite;
  width: 50px;
  height: 50px;
}
</style>
