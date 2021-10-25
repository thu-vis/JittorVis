'use strict';
const path = require('path');
const utils = require('./utils');
const webpack = require('webpack');
const config = require('../config');
const merge = require('webpack-merge');
const baseWebpackConfig = require('./webpack.base.conf');
const CopyWebpackPlugin = require('copy-webpack-plugin');
const HtmlWebpackPlugin = require('html-webpack-plugin');
const {VueLoaderPlugin} = require('vue-loader');

const env = require('../config/prod.env');

const webpackConfig = merge(baseWebpackConfig, {
    mode: 'production',
    module: {
        rules: utils.styleLoaders({sourceMap: config.dev.cssSourceMap, usePostCSS: true}),
    },
    devtool: 'eval-source-map',
    output: {
        path: config.build.assetsRoot,
        filename: utils.assetsPath('js/[name].[chunkhash].js'),
        chunkFilename: utils.assetsPath('js/[id].[chunkhash].js'),
        clean: false,
    },
    optimization: {
        splitChunks: {
            chunks: 'all',
            maxSize: 500000,
        },
    },
    plugins: [
    // http://vuejs.github.io/vue-loader/en/workflow/production.html
        new webpack.DefinePlugin({
            'process.env': env,
        }),

        // copy custom static assets
        new CopyWebpackPlugin([
            {
                from: path.resolve(__dirname, '../static'),
                to: config.build.assetsSubDirectory,
                ignore: ['.*'],
            },
        ]),
        new HtmlWebpackPlugin({
            title: 'JittorVis',
            filename: 'templates/index.html',
            template: 'index.html',
        }),
        new VueLoaderPlugin(),
    ],
    resolve: {
        alias: {
            'vue$': 'vue/dist/vue.esm.js', // 'vue/dist/vue.common.js' for webpack 1
        },
    },
});

module.exports = webpackConfig;
