const devWebpackConfig = require('./build/webpack.dev.conf.js');
devWebpackConfig.devServer.port = 6060;
// ./styleguide.config.js
module.exports = {
    webpackConfig: devWebpackConfig,
    styleguideDir: '../doc/frontend',
};
