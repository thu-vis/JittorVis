<template>
    <div id="data-content">
        <div id="left">
            <div id="confusion-matrix-container">
                <div class="dummy-container">
                    <confusion-matrix id="confusion-matrix"></confusion-matrix>
                </div>
            </div>
        </div>
        <div id="right">

        </div>
    </div>
</template>

<script>
import ConfusionMatrix from './ConfusionMatrix.vue';

export default {
    components: {ConfusionMatrix},
    name: 'DataView',
};
</script>

<style scoped>
#data-content {
    width: 100%;
    height: 100%;
    overflow: hidden;
    display: flex;
}

#left {
    width: 50%;
    height: 100%;
}

#right {
    width: 50%;
    height: 100%;
}

#confusion-matrix-container {
  position: relative;
  width: 100%;
  padding-top: 100%;
}

.dummy-container {
  position:  absolute;
  top: 0;
  left: 0;
  bottom: 0;
  right: 0;
  width: 100%;
  height: 100%;
  display: flex;
  justify-content: flex-start;
  align-items: flex-start;
}
</style>
