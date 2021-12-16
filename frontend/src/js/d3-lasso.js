/* eslint-disable */
(function (global, factory) {
    typeof exports === 'object' && typeof module !== 'undefined' ? factory(exports, require('d3-selection'), require('d3-drag')) :
    typeof define === 'function' && define.amd ? define(['exports', 'd3-selection', 'd3-drag'], factory) :
    (factory((global.d3 = global.d3 || {}),global.d3,global.d3));
}(window, function (exports,d3Selection,d3Drag) { 'use strict';

    function createCommonjsModule(fn, module) {
    	return module = { exports: {} }, fn(module, module.exports), module.exports;
    }

    var __moduleExports$1 = createCommonjsModule(function (module) {
    "use strict"

    module.exports = twoProduct

    var SPLITTER = +(Math.pow(2, 27) + 1.0)

    function twoProduct(a, b, result) {
      var x = a * b

      var c = SPLITTER * a
      var abig = c - a
      var ahi = c - abig
      var alo = a - ahi

      var d = SPLITTER * b
      var bbig = d - b
      var bhi = d - bbig
      var blo = b - bhi

      var err1 = x - (ahi * bhi)
      var err2 = err1 - (alo * bhi)
      var err3 = err2 - (ahi * blo)

      var y = alo * blo - err3

      if(result) {
        result[0] = y
        result[1] = x
        return result
      }

      return [ y, x ]
    }
    });

    var __moduleExports$2 = createCommonjsModule(function (module) {
    "use strict"

    module.exports = linearExpansionSum

    //Easy case: Add two scalars
    function scalarScalar(a, b) {
      var x = a + b
      var bv = x - a
      var av = x - bv
      var br = b - bv
      var ar = a - av
      var y = ar + br
      if(y) {
        return [y, x]
      }
      return [x]
    }

    function linearExpansionSum(e, f) {
      var ne = e.length|0
      var nf = f.length|0
      if(ne === 1 && nf === 1) {
        return scalarScalar(e[0], f[0])
      }
      var n = ne + nf
      var g = new Array(n)
      var count = 0
      var eptr = 0
      var fptr = 0
      var abs = Math.abs
      var ei = e[eptr]
      var ea = abs(ei)
      var fi = f[fptr]
      var fa = abs(fi)
      var a, b
      if(ea < fa) {
        b = ei
        eptr += 1
        if(eptr < ne) {
          ei = e[eptr]
          ea = abs(ei)
        }
      } else {
        b = fi
        fptr += 1
        if(fptr < nf) {
          fi = f[fptr]
          fa = abs(fi)
        }
      }
      if((eptr < ne && ea < fa) || (fptr >= nf)) {
        a = ei
        eptr += 1
        if(eptr < ne) {
          ei = e[eptr]
          ea = abs(ei)
        }
      } else {
        a = fi
        fptr += 1
        if(fptr < nf) {
          fi = f[fptr]
          fa = abs(fi)
        }
      }
      var x = a + b
      var bv = x - a
      var y = b - bv
      var q0 = y
      var q1 = x
      var _x, _bv, _av, _br, _ar
      while(eptr < ne && fptr < nf) {
        if(ea < fa) {
          a = ei
          eptr += 1
          if(eptr < ne) {
            ei = e[eptr]
            ea = abs(ei)
          }
        } else {
          a = fi
          fptr += 1
          if(fptr < nf) {
            fi = f[fptr]
            fa = abs(fi)
          }
        }
        b = q0
        x = a + b
        bv = x - a
        y = b - bv
        if(y) {
          g[count++] = y
        }
        _x = q1 + x
        _bv = _x - q1
        _av = _x - _bv
        _br = x - _bv
        _ar = q1 - _av
        q0 = _ar + _br
        q1 = _x
      }
      while(eptr < ne) {
        a = ei
        b = q0
        x = a + b
        bv = x - a
        y = b - bv
        if(y) {
          g[count++] = y
        }
        _x = q1 + x
        _bv = _x - q1
        _av = _x - _bv
        _br = x - _bv
        _ar = q1 - _av
        q0 = _ar + _br
        q1 = _x
        eptr += 1
        if(eptr < ne) {
          ei = e[eptr]
        }
      }
      while(fptr < nf) {
        a = fi
        b = q0
        x = a + b
        bv = x - a
        y = b - bv
        if(y) {
          g[count++] = y
        } 
        _x = q1 + x
        _bv = _x - q1
        _av = _x - _bv
        _br = x - _bv
        _ar = q1 - _av
        q0 = _ar + _br
        q1 = _x
        fptr += 1
        if(fptr < nf) {
          fi = f[fptr]
        }
      }
      if(q0) {
        g[count++] = q0
      }
      if(q1) {
        g[count++] = q1
      }
      if(!count) {
        g[count++] = 0.0  
      }
      g.length = count
      return g
    }
    });

    var __moduleExports$4 = createCommonjsModule(function (module) {
    "use strict"

    module.exports = fastTwoSum

    function fastTwoSum(a, b, result) {
    	var x = a + b
    	var bv = x - a
    	var av = x - bv
    	var br = b - bv
    	var ar = a - av
    	if(result) {
    		result[0] = ar + br
    		result[1] = x
    		return result
    	}
    	return [ar+br, x]
    }
    });

    var __moduleExports$3 = createCommonjsModule(function (module) {
    "use strict"

    var twoProduct = __moduleExports$1
    var twoSum = __moduleExports$4

    module.exports = scaleLinearExpansion

    function scaleLinearExpansion(e, scale) {
      var n = e.length
      if(n === 1) {
        var ts = twoProduct(e[0], scale)
        if(ts[0]) {
          return ts
        }
        return [ ts[1] ]
      }
      var g = new Array(2 * n)
      var q = [0.1, 0.1]
      var t = [0.1, 0.1]
      var count = 0
      twoProduct(e[0], scale, q)
      if(q[0]) {
        g[count++] = q[0]
      }
      for(var i=1; i<n; ++i) {
        twoProduct(e[i], scale, t)
        var pq = q[1]
        twoSum(pq, t[0], q)
        if(q[0]) {
          g[count++] = q[0]
        }
        var a = t[1]
        var b = q[1]
        var x = a + b
        var bv = x - a
        var y = b - bv
        q[1] = x
        if(y) {
          g[count++] = y
        }
      }
      if(q[1]) {
        g[count++] = q[1]
      }
      if(count === 0) {
        g[count++] = 0.0
      }
      g.length = count
      return g
    }
    });

    var __moduleExports$5 = createCommonjsModule(function (module) {
    "use strict"

    module.exports = robustSubtract

    //Easy case: Add two scalars
    function scalarScalar(a, b) {
      var x = a + b
      var bv = x - a
      var av = x - bv
      var br = b - bv
      var ar = a - av
      var y = ar + br
      if(y) {
        return [y, x]
      }
      return [x]
    }

    function robustSubtract(e, f) {
      var ne = e.length|0
      var nf = f.length|0
      if(ne === 1 && nf === 1) {
        return scalarScalar(e[0], -f[0])
      }
      var n = ne + nf
      var g = new Array(n)
      var count = 0
      var eptr = 0
      var fptr = 0
      var abs = Math.abs
      var ei = e[eptr]
      var ea = abs(ei)
      var fi = -f[fptr]
      var fa = abs(fi)
      var a, b
      if(ea < fa) {
        b = ei
        eptr += 1
        if(eptr < ne) {
          ei = e[eptr]
          ea = abs(ei)
        }
      } else {
        b = fi
        fptr += 1
        if(fptr < nf) {
          fi = -f[fptr]
          fa = abs(fi)
        }
      }
      if((eptr < ne && ea < fa) || (fptr >= nf)) {
        a = ei
        eptr += 1
        if(eptr < ne) {
          ei = e[eptr]
          ea = abs(ei)
        }
      } else {
        a = fi
        fptr += 1
        if(fptr < nf) {
          fi = -f[fptr]
          fa = abs(fi)
        }
      }
      var x = a + b
      var bv = x - a
      var y = b - bv
      var q0 = y
      var q1 = x
      var _x, _bv, _av, _br, _ar
      while(eptr < ne && fptr < nf) {
        if(ea < fa) {
          a = ei
          eptr += 1
          if(eptr < ne) {
            ei = e[eptr]
            ea = abs(ei)
          }
        } else {
          a = fi
          fptr += 1
          if(fptr < nf) {
            fi = -f[fptr]
            fa = abs(fi)
          }
        }
        b = q0
        x = a + b
        bv = x - a
        y = b - bv
        if(y) {
          g[count++] = y
        }
        _x = q1 + x
        _bv = _x - q1
        _av = _x - _bv
        _br = x - _bv
        _ar = q1 - _av
        q0 = _ar + _br
        q1 = _x
      }
      while(eptr < ne) {
        a = ei
        b = q0
        x = a + b
        bv = x - a
        y = b - bv
        if(y) {
          g[count++] = y
        }
        _x = q1 + x
        _bv = _x - q1
        _av = _x - _bv
        _br = x - _bv
        _ar = q1 - _av
        q0 = _ar + _br
        q1 = _x
        eptr += 1
        if(eptr < ne) {
          ei = e[eptr]
        }
      }
      while(fptr < nf) {
        a = fi
        b = q0
        x = a + b
        bv = x - a
        y = b - bv
        if(y) {
          g[count++] = y
        } 
        _x = q1 + x
        _bv = _x - q1
        _av = _x - _bv
        _br = x - _bv
        _ar = q1 - _av
        q0 = _ar + _br
        q1 = _x
        fptr += 1
        if(fptr < nf) {
          fi = -f[fptr]
        }
      }
      if(q0) {
        g[count++] = q0
      }
      if(q1) {
        g[count++] = q1
      }
      if(!count) {
        g[count++] = 0.0  
      }
      g.length = count
      return g
    }
    });

    var __moduleExports = createCommonjsModule(function (module) {
    "use strict"

    var twoProduct = __moduleExports$1
    var robustSum = __moduleExports$2
    var robustScale = __moduleExports$3
    var robustSubtract = __moduleExports$5

    var NUM_EXPAND = 5

    var EPSILON     = 1.1102230246251565e-16
    var ERRBOUND3   = (3.0 + 16.0 * EPSILON) * EPSILON
    var ERRBOUND4   = (7.0 + 56.0 * EPSILON) * EPSILON

    function cofactor(m, c) {
      var result = new Array(m.length-1)
      for(var i=1; i<m.length; ++i) {
        var r = result[i-1] = new Array(m.length-1)
        for(var j=0,k=0; j<m.length; ++j) {
          if(j === c) {
            continue
          }
          r[k++] = m[i][j]
        }
      }
      return result
    }

    function matrix(n) {
      var result = new Array(n)
      for(var i=0; i<n; ++i) {
        result[i] = new Array(n)
        for(var j=0; j<n; ++j) {
          result[i][j] = ["m", j, "[", (n-i-1), "]"].join("")
        }
      }
      return result
    }

    function sign(n) {
      if(n & 1) {
        return "-"
      }
      return ""
    }

    function generateSum(expr) {
      if(expr.length === 1) {
        return expr[0]
      } else if(expr.length === 2) {
        return ["sum(", expr[0], ",", expr[1], ")"].join("")
      } else {
        var m = expr.length>>1
        return ["sum(", generateSum(expr.slice(0, m)), ",", generateSum(expr.slice(m)), ")"].join("")
      }
    }

    function determinant(m) {
      if(m.length === 2) {
        return [["sum(prod(", m[0][0], ",", m[1][1], "),prod(-", m[0][1], ",", m[1][0], "))"].join("")]
      } else {
        var expr = []
        for(var i=0; i<m.length; ++i) {
          expr.push(["scale(", generateSum(determinant(cofactor(m, i))), ",", sign(i), m[0][i], ")"].join(""))
        }
        return expr
      }
    }

    function orientation(n) {
      var pos = []
      var neg = []
      var m = matrix(n)
      var args = []
      for(var i=0; i<n; ++i) {
        if((i&1)===0) {
          pos.push.apply(pos, determinant(cofactor(m, i)))
        } else {
          neg.push.apply(neg, determinant(cofactor(m, i)))
        }
        args.push("m" + i)
      }
      var posExpr = generateSum(pos)
      var negExpr = generateSum(neg)
      var funcName = "orientation" + n + "Exact"
      var code = ["function ", funcName, "(", args.join(), "){var p=", posExpr, ",n=", negExpr, ",d=sub(p,n);\
return d[d.length-1];};return ", funcName].join("")
      var proc = new Function("sum", "prod", "scale", "sub", code)
      return proc(robustSum, twoProduct, robustScale, robustSubtract)
    }

    var orientation3Exact = orientation(3)
    var orientation4Exact = orientation(4)

    var CACHED = [
      function orientation0() { return 0 },
      function orientation1() { return 0 },
      function orientation2(a, b) { 
        return b[0] - a[0]
      },
      function orientation3(a, b, c) {
        var l = (a[1] - c[1]) * (b[0] - c[0])
        var r = (a[0] - c[0]) * (b[1] - c[1])
        var det = l - r
        var s
        if(l > 0) {
          if(r <= 0) {
            return det
          } else {
            s = l + r
          }
        } else if(l < 0) {
          if(r >= 0) {
            return det
          } else {
            s = -(l + r)
          }
        } else {
          return det
        }
        var tol = ERRBOUND3 * s
        if(det >= tol || det <= -tol) {
          return det
        }
        return orientation3Exact(a, b, c)
      },
      function orientation4(a,b,c,d) {
        var adx = a[0] - d[0]
        var bdx = b[0] - d[0]
        var cdx = c[0] - d[0]
        var ady = a[1] - d[1]
        var bdy = b[1] - d[1]
        var cdy = c[1] - d[1]
        var adz = a[2] - d[2]
        var bdz = b[2] - d[2]
        var cdz = c[2] - d[2]
        var bdxcdy = bdx * cdy
        var cdxbdy = cdx * bdy
        var cdxady = cdx * ady
        var adxcdy = adx * cdy
        var adxbdy = adx * bdy
        var bdxady = bdx * ady
        var det = adz * (bdxcdy - cdxbdy) 
                + bdz * (cdxady - adxcdy)
                + cdz * (adxbdy - bdxady)
        var permanent = (Math.abs(bdxcdy) + Math.abs(cdxbdy)) * Math.abs(adz)
                      + (Math.abs(cdxady) + Math.abs(adxcdy)) * Math.abs(bdz)
                      + (Math.abs(adxbdy) + Math.abs(bdxady)) * Math.abs(cdz)
        var tol = ERRBOUND4 * permanent
        if ((det > tol) || (-det > tol)) {
          return det
        }
        return orientation4Exact(a,b,c,d)
      }
    ]

    function slowOrient(args) {
      var proc = CACHED[args.length]
      if(!proc) {
        proc = CACHED[args.length] = orientation(args.length)
      }
      return proc.apply(undefined, args)
    }

    function generateOrientationProc() {
      while(CACHED.length <= NUM_EXPAND) {
        CACHED.push(orientation(CACHED.length))
      }
      var args = []
      var procArgs = ["slow"]
      for(var i=0; i<=NUM_EXPAND; ++i) {
        args.push("a" + i)
        procArgs.push("o" + i)
      }
      var code = [
        "function getOrientation(", args.join(), "){switch(arguments.length){case 0:case 1:return 0;"
      ]
      for(var i=2; i<=NUM_EXPAND; ++i) {
        code.push("case ", i, ":return o", i, "(", args.slice(0, i).join(), ");")
      }
      code.push("}var s=new Array(arguments.length);for(var i=0;i<arguments.length;++i){s[i]=arguments[i]};return slow(s);}return getOrientation")
      procArgs.push(code.join(""))

      var proc = Function.apply(undefined, procArgs)
      module.exports = proc.apply(undefined, [slowOrient].concat(CACHED))
      for(var i=0; i<=NUM_EXPAND; ++i) {
        module.exports[i] = CACHED[i]
      }
    }

    generateOrientationProc()
    });

    var robustPnp = createCommonjsModule(function (module) {
    module.exports = robustPointInPolygon

    var orient = __moduleExports

    function robustPointInPolygon(vs, point) {
      var x = point[0]
      var y = point[1]
      var n = vs.length
      var inside = 1
      var lim = n
      for(var i = 0, j = n-1; i<lim; j=i++) {
        var a = vs[i]
        var b = vs[j]
        var yi = a[1]
        var yj = b[1]
        if(yj < yi) {
          if(yj < y && y < yi) {
            var s = orient(a, b, point)
            if(s === 0) {
              return 0
            } else {
              inside ^= (0 < s)|0
            }
          } else if(y === yi) {
            var c = vs[(i+1)%n]
            var yk = c[1]
            if(yi < yk) {
              var s = orient(a, b, point)
              if(s === 0) {
                return 0
              } else {
                inside ^= (0 < s)|0
              }
            }
          }
        } else if(yi < yj) {
          if(yi < y && y < yj) {
            var s = orient(a, b, point)
            if(s === 0) {
              return 0
            } else {
              inside ^= (s < 0)|0
            }
          } else if(y === yi) {
            var c = vs[(i+1)%n]
            var yk = c[1]
            if(yk < yi) {
              var s = orient(a, b, point)
              if(s === 0) {
                return 0
              } else {
                inside ^= (s < 0)|0
              }
            }
          }
        } else if(y === yi) {
          var x0 = Math.min(a[0], b[0])
          var x1 = Math.max(a[0], b[0])
          if(i === 0) {
            while(j>0) {
              var k = (j+n-1)%n
              var p = vs[k]
              if(p[1] !== y) {
                break
              }
              var px = p[0]
              x0 = Math.min(x0, px)
              x1 = Math.max(x1, px)
              j = k
            }
            if(j === 0) {
              if(x0 <= x && x <= x1) {
                return 0
              }
              return 1 
            }
            lim = j+1
          }
          var y0 = vs[(j+n-1)%n][1]
          while(i+1<lim) {
            var p = vs[i+1]
            if(p[1] !== y) {
              break
            }
            var px = p[0]
            x0 = Math.min(x0, px)
            x1 = Math.max(x1, px)
            i += 1
          }
          if(x0 <= x && x <= x1) {
            return 0
          }
          var y1 = vs[(i+1)%n][1]
          if(x < x0 && (y0 < y !== y1 < y)) {
            inside ^= 1
          }
        }
      }
      return 2 * inside - 1
    }
    });

    function lasso() {

        var items =[],
            closePathDistance = 75,
            closePathSelect = true,
            isPathClosed = false,
            hoverSelect = true,
            targetArea,
            on = {start:function(){}, draw: function(){}, end: function(){}};

        // Function to execute on call
        function lasso(_this) {

            // add a new group for the lasso
            var g = _this.append("g")
                .attr("class","lasso");
            
            // add the drawn path for the lasso
            var dyn_path = g.append("path")
                .attr("class","drawn");
            
            // add a closed path
            var close_path = g.append("path")
                .attr("class","loop_close");
            
            // add an origin node
            var origin_node = g.append("circle")
                .attr("class","origin");

            // The transformed lasso path for rendering
            var tpath;

            // The lasso origin for calculations
            var origin;

            // The transformed lasso origin for rendering
            var torigin;

            // Store off coordinates drawn
            var drawnCoords;

             // Apply drag behaviors
            var drag = d3.drag()
                .on("start",dragstart)
                .on("drag",dragmove)
                .on("end",dragend);

            // Call drag
            targetArea.call(drag);

            function dragstart() {
                // Init coordinates
                drawnCoords = [];

                // Initialize paths
                tpath = "";
                dyn_path.attr("d",null);
                close_path.attr("d",null);

                // Set every item to have a false selection and reset their center point and counters
                items.nodes().forEach(function(e) {            
                    e.__lasso.possible = false;
                    e.__lasso.selected = false;
                    e.__lasso.hoverSelect = false;
                    e.__lasso.loopSelect = false;
                    
                    var box = e.getBoundingClientRect();
                    e.__lasso.lassoPoint = [Math.round(box.left + box.width/2),Math.round(box.top + box.height/2)];
                });

                // if hover is on, add hover function
                if(hoverSelect) {
                    items.on("mouseover.lasso",function() {
                        // if hovered, change lasso selection attribute to true
                        this.__lasso.hoverSelect = true;
                    });
                }

                // Run user defined start function
                on.start();
            }

            function dragmove(e) {
                // Get mouse position within body, used for calculations
                var x,y;
                if(e.sourceEvent.type === "touchmove") {
                    x = e.sourceEvent.touches[0].clientX;
                    y = e.sourceEvent.touches[0].clientY;
                }
                else {
                    x = e.sourceEvent.clientX;
                    y = e.sourceEvent.clientY;
                }
                

                // Get mouse position within drawing area, used for rendering
                var tx = e.sourceEvent.offsetX;
                var ty = e.sourceEvent.offsetY;

                // Initialize the path or add the latest point to it
                if (tpath==="") {
                    tpath = tpath + "M " + tx + " " + ty;
                    origin = [x,y];
                    torigin = [tx,ty];
                    // Draw origin node
                    origin_node
                        .attr("cx",tx)
                        .attr("cy",ty)
                        .attr("r",7)
                        .attr("display",null);
                }
                else {
                    tpath = tpath + " L " + tx + " " + ty;
                }

                drawnCoords.push([x,y]);

                // Calculate the current distance from the lasso origin
                var distance = Math.sqrt(Math.pow(x-origin[0],2)+Math.pow(y-origin[1],2));

                // Set the closed path line
                var close_draw_path = "M " + tx + " " + ty + " L " + torigin[0] + " " + torigin[1];

                // Draw the lines
                dyn_path.attr("d",tpath);

                close_path.attr("d",close_draw_path);

                // Check if the path is closed
                isPathClosed = distance<=closePathDistance ? true : false;

                // If within the closed path distance parameter, show the closed path. otherwise, hide it
                if(isPathClosed && closePathSelect) {
                    close_path.attr("display",null);
                }
                else {
                    close_path.attr("display","none");
                }

                items.nodes().forEach(function(n) {
                    n.__lasso.loopSelect = (isPathClosed && closePathSelect) ? (robustPnp(drawnCoords,n.__lasso.lassoPoint) < 1) : false; 
                    n.__lasso.possible = n.__lasso.hoverSelect || n.__lasso.loopSelect; 
                });

                on.draw();
            }

            function dragend() {
                // Remove mouseover tagging function
                items.on("mouseover.lasso",null);

                items.nodes().forEach(function(n) {
                    n.__lasso.selected = n.__lasso.possible;
                    n.__lasso.possible = false;
                });

                // Clear lasso
                dyn_path.attr("d",null);
                close_path.attr("d",null);
                origin_node.attr("display","none");

                // Run user defined end function
                on.end();
            }
        }

        // Set or get list of items for lasso to select
        lasso.items  = function(_) {
            if (!arguments.length) return items;
            items = _;
            var nodes = items.nodes();
            nodes.forEach(function(n) {
                n.__lasso = {
                    "possible": false,
                    "selected": false
                };
            });
            return lasso;
        };

        // Return possible items
        lasso.possibleItems = function() {
            return items.filter(function() {
                return this.__lasso.possible;
            });
        }

        // Return selected items
        lasso.selectedItems = function() {
            return items.filter(function() {
                return this.__lasso.selected;
            });
        }

        // Return not possible items
        lasso.notPossibleItems = function() {
            return items.filter(function() {
                return !this.__lasso.possible;
            });
        }

        // Return not selected items
        lasso.notSelectedItems = function() {
            return items.filter(function() {
                return !this.__lasso.selected;
            });
        }

        // Distance required before path auto closes loop
        lasso.closePathDistance  = function(_) {
            if (!arguments.length) return closePathDistance;
            closePathDistance = _;
            return lasso;
        };

        // Option to loop select or not
        lasso.closePathSelect = function(_) {
            if (!arguments.length) return closePathSelect;
            closePathSelect = _===true ? true : false;
            return lasso;
        };

        // Not sure what this is for
        lasso.isPathClosed = function(_) {
            if (!arguments.length) return isPathClosed;
            isPathClosed = _===true ? true : false;
            return lasso;
        };

        // Option to select on hover or not
        lasso.hoverSelect = function(_) {
            if (!arguments.length) return hoverSelect;
            hoverSelect = _===true ? true : false;
            return lasso;
        };

        // Events
        lasso.on = function(type,_) {
            if(!arguments.length) return on;
            if(arguments.length===1) return on[type];
            var types = ["start","draw","end"];
            if(types.indexOf(type)>-1) {
                on[type] = _;
            }
            return lasso;
        };

        // Area where lasso can be triggered from
        lasso.targetArea = function(_) {
            if(!arguments.length) return targetArea;
            targetArea = _;
            return lasso;
        }


        
        return lasso;
    };

    exports.lasso = lasso;

    Object.defineProperty(exports, '__esModule', { value: true });

}));