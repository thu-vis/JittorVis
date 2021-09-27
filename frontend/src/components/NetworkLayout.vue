<template>
    <text>this is layout</text>
</template>

<script>
import { mapGetters, mapState } from 'vuex'

export default {
    name: 'network-layout',
    computed: {
        ...mapGetters([
            'network'
        ]),
        ...mapState([
            'focusID'
        ])
    }
}
</script>
