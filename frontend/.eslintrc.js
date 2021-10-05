module.exports = {
    'env': {
        'browser': true,
        'es2021': true,
    },
    'extends': [
        'plugin:vue/essential',
        'google',
    ],
    'parserOptions': {
        'ecmaVersion': 12,
        'sourceType': 'module',
    },
    'plugins': [
        'vue',
    ],
    'rules': {
    // allow async-await
        'generator-star-spacing': 'off',
        // allow debugger during development
        'no-debugger': process.env.NODE_ENV === 'production' ? 'error' : 'off',
        'indent': ['error', 4],
        'max-len': [
            'error',
            {'code': 150},
        ],
    },
};
