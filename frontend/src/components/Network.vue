<template>
    <p>{{ APIBASE }}</p>
</template>

<script>
import { mapState } from 'vuex'

export default {
    name: 'network',
    computed: {
        ...mapState([
            'APIBASE'
        ])
    }
}
</script>
