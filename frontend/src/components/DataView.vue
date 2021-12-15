<template>
    <div id="data-content">
        <div id="left">
            <div id="left-confusion-matrix-container">
                <confusion-matrix id="confusion-matrix"></confusion-matrix>
            </div>
        </div>
        <div id="right">
            <gird-layout></gird-layout>
        </div>
    </div>
</template>

<script>
import ConfusionMatrix from './ConfusionMatrix.vue';
import GirdLayout from './GridLayout.vue';

export default {
    components: {ConfusionMatrix, GirdLayout},
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

#left-confusion-matrix-container {
  width: 100%;
  height: 100%;
  display: flex;
  margin: 10px 10px 10px 10px;
}
</style>
