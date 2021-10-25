const {staticWebpackConfig} = require('./build/webpack.dev.conf.js');

devWebpackConfig = staticWebpackConfig;
devWebpackConfig.devServer.port = 6060;
console.log(devWebpackConfig);
// ./styleguide.config.js
module.exports = {
    webpackConfig: devWebpackConfig,
    styleguideDir: '../doc/frontend',
};
