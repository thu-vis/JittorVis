<template>
<svg id="navigation-tree">
    <text>this is navigation</text>
</svg>
</template>

<script>
import {mapGetters, mapState} from 'vuex';

export default {
    name: 'navigation',
    computed: {
        ...mapGetters([
            'network',
        ]),
        ...mapState([
            'focusID',
        ]),
    },
};
</script>

<style scoped>
#navigation-tree {
  width: 80%;
  height: 80%;
}
</style>
