/**
 *
 * Neural class handles all "communication" between the user and the neural net
 *
 */
var Neural = function() {
    var self = this;

    this.settings = {
        runTraining: false,
        displayStatistics: false,
        result: [
            [1], [0], [0], [0], [0], [0], [0], [0], [0], [0]
        ],
    };

    // timeouts / intervals
    this.updateTimeout = -1;
    this.trainingInterval = -1;

    // 
    this.mouse = {x:0, y:0};
    this.tile = {x:0, y:0};
    this.numTiles = {x:32, y:32};
    this.buttonsPressed = [];
    this.tileSize = 16;
    this.numTiles = {x:32, y:32};
    this.data = [];
    this.trainingData = [];

    this.isTraining = false;

    // neural net
    this.network = new Network([
        {size: {x:32, y:32}, radius: -2},
        {size: {x:16, y:16}, radius: -2},
        {size: {x:8,  y:8},  radius: true},
        {size: {x:10, y:1},  radius: false},
    ]);
    // console.log(this.network);

    this.drawCanvas = document.getElementById('draw');
    this.drawCanvas.width = this.numTiles.x * this.tileSize;
    this.drawCanvas.height = this.numTiles.y * this.tileSize;
    this.drawCtx = this.drawCanvas.getContext('2d');
    this.drawCanvasOffset = jQuery(this.drawCanvas).position();

    self.clearCanvas = function() {
        for (var x = 0; x < this.numTiles.x; x++) {
            this.data[x] = [];
            for (var y = 0; y < this.numTiles.y; y++) {
                this.data[x][y] = 0;
            }
        }
    }

    this.start = function() {

        this.clearCanvas();

        // bind stuff on elements
        jQuery('#settingsForm input, #settingsForm select').on('change', function(e) { self.handleSettingChange(e, jQuery(this)); });
        jQuery('#clearCanvas').on('click', function(e) { self.clearCanvas(); });
        jQuery('#saveTraining').on('click', function(e) {
            self.trainingData.push({
                data: self.data.slice(),
                result: self.settings.result.slice(),
            });
            console.log(self.trainingData);
            // self.clearCanvas(); 
        });
        jQuery('canvas').bind('contextmenu', function(e) { return false; });

        // bind mouse stuff
        window.onmousemove = function(e) { self.handleMouseMove(e); };
        jQuery(window).mousedown(function(e) { self.mousedown(e); });
        jQuery(window).mouseup(function(e) { self.mouseup(e); });

        // training
        this.trainingInterval = setInterval(this.train, 1000/5);

        // start drawing
        requestAnimFrame(this.draw);
    }

    this.train = function() {
        if(self.isTraining) return;
        self.isTraining = true;
        
        // self.trainingData;
        // console.log("train");

        for (var i = 0; i < self.trainingData.length; i++) {
            self.network.feedForward(self.trainingData[i].data);
            self.network.backPropagation(self.trainingData[i].result);
        }
        self.network.feedForward(self.data);
        jQuery('#prediction').html(self.network.getReadableResult());
        

        if(self.settings.displayStatistics) {
            jQuery('#statistics').html(
                'Error: '+self.network.error+'<br>'+
                'Recent average error: '+self.network.recentAverageError+'<br>'+
                'Ran # of training: '+i
            );
        }

        self.isTraining = false;
    }

    this.handleSettingChange = function(event, that) {
        var value = 0;
        if(that.attr('type') == 'checkbox') {
            value = that.prop('checked');
        } else if(that.is('select')) {
            value = [];
            for (var i = 0; i < 10; i++) {
                value[i] = [i == that.val() ? 1 : 0];
            }
        } else {
            value = that.val();
        }
        this.updateSetting(that.attr('name'), value);
    }


    this.updateSetting = function(name, value) {
        this.settings[name] = value;
        console.log(this.settings);
    }

    this.draw = function() {
        requestAnimFrame(self.draw);
        self.drawCtx.fillStyle = "rgba(255,255,255,1)";
        self.drawCtx.fillRect(0, 0, self.drawCanvas.width, self.drawCanvas.height);
        self.drawCtx.fillStyle = "rgba(0,0,0,1)";
        for (var x = 0; x < self.numTiles.x; x++) {
            for (var y = 0; y < self.numTiles.y; y++) {
                if(self.data[x][y] == 1) {
                    self.drawCtx.fillRect(x * self.tileSize, y * self.tileSize, self.tileSize, self.tileSize);
                }
            }
        }
    }

    this.drawTile = function() {
        if(this.isOnCanvas()) {
            if(this.buttonsPressed[1]) {
                self.data[this.tile.x][this.tile.y] = 1;
                window.clearTimeout(this.updateTimeout);
                this.updateTimeout = window.setTimeout(function() {
                    self.network.feedForward(self.data);
                    jQuery('#prediction').html(self.network.getReadableResult());
                }, 1000);
            } else if(this.buttonsPressed[3]) {
                self.data[this.tile.x][this.tile.y] = 0;
            }
        } else {
            // this.buttonsPressed[1] = this.buttonsPressed[3] = false;
        }
    }

    this.handleMouseMove = function(e) {
        var event = window.event || e;
        this.mouse.x = event.clientX;
        this.mouse.y = event.clientY;
        this.tile.x = Math.floor((this.mouse.x - this.drawCanvasOffset.left) / this.tileSize);
        this.tile.y = Math.floor((this.mouse.y - this.drawCanvasOffset.top) / this.tileSize);
        this.drawTile();
    }

    this.mousedown = function(e) {
        var evnt = window.event || e;
        this.buttonsPressed[ evnt.which ] = true;
        this.drawTile();
    }

    this.mouseup = function(e) {
        var evnt = window.event || e;
        this.buttonsPressed[ evnt.which ] = false;
        this.drawTile();
    }

    this.isOnCanvas = function() {
        return  this.mouse.x > this.drawCanvasOffset.left &&
                this.mouse.x < this.drawCanvasOffset.left + this.drawCanvas.width &&
                this.mouse.y > this.drawCanvasOffset.top &&
                this.mouse.y < this.drawCanvasOffset.top + this.drawCanvas.height;
    }

};

// shim layer with setTimeout fallback
window.requestAnimFrame = (function(){
    return window.requestAnimationFrame ||
        window.webkitRequestAnimationFrame  ||
        window.mozRequestAnimationFrame     ||
        function( callback ){
            window.setTimeout(callback, 1000 / 60);
        };
})();