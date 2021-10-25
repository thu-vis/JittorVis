<template>
<svg id="navigation-tree">
    <g class="container" v-if="layoutNetwork">
        <g class="links"></g>
        <g class="nodes"></g>
    </g>
    <g v-else>
        <text fill="gray" x="50%" y="50%" text-anchor="middle">数据不可用</text>
    </g>
</svg>
</template>

<script>
/* eslint-disable */
import { mapGetters, mapState } from 'vuex'
import * as d3 from 'd3'
import GlobalVar from './GlovalVar.vue';

export default {
    name: 'navigation',
    computed: {
        ...mapGetters([
            'layoutInfo',
        ]),
        layoutNetwork: function() {
            return this.layoutInfo.layoutNetwork;
        },
        focus_ID: function() {
            return this.layoutInfo.focusID || (this.root && this.root.data.name);
        }
    },
    mixins: [GlobalVar],
    props:{
        canFocusNode: {
            type: Boolean,
            default: false
        }
    },
    data(){
        return {
            root:undefined,
            ordering:{},
        }
    },
    watch:{
        'layoutInfo.layoutNetwork': function(val, oldval) {
            if(Object.keys(oldval).length===0){
                this.initTree();
            }
            else {
                this.updateTree();
            }
        },
        'layoutInfo.t': function(val) {
            this.updateTree();
        }
    },
    methods:{
        initTree()
        {
            // calc ordering
            // console.log("navigation ready",this.canFocusNode)
            let items={}
            let ordering=[]
            let n=0
            this.ordering={}
            for(let key in this.layoutNetwork)
            {
                if(this.layoutNetwork[key].children.length==0)
                {
                    items[key]=this.layoutNetwork[key].inputs.length
                    if(this.layoutNetwork[key].inputs.length==0)ordering.push(key)
                    n++
                }
            }
            for(let i=0;i<n;i++)
            {
                const key=ordering[i]
                this.ordering[key]=i+1
                if(i>=ordering.length)return false
                for(let j in this.layoutNetwork[key].outputs)
                {
                    const chkey=this.layoutNetwork[key].outputs[j]
                    if(--items[chkey]==0)ordering.push(chkey)
                }
            }
            const that=this
            const calcMiddle=(key)=>{
                if(that.ordering[key])return that.ordering[key]
                let ans=0
                for(let ch in that.layoutNetwork[key].children){
                    if(calcMiddle(that.layoutNetwork[key].children[ch]) >ans)
                        ans=that.ordering[that.layoutNetwork[key].children[ch]]
                }
                that.ordering[key]=ans
                return ans
            }
            for(let key in this.layoutNetwork)calcMiddle(key)

            // building tree_list
            window.d3=d3
            window.layoutNetwork=this.layoutNetwork
            let network_list=[]
            for (let key in this.layoutNetwork)
            {
                const id=this.layoutNetwork[key].id
                const fa=this.layoutNetwork[key].parent
                network_list.push({
                    "name":id,
                    "parent":fa,
                    //"attrs":this.layoutNetwork[key].attrs,
                    "type":this.layoutNetwork[key].type,
                    "ordering":this.ordering[id]
                })
            }
            network_list.sort((i,j)=>{
                return i.ordering-j.ordering
            })
            window.layoutNetwork_list=network_list
            const root = d3.stratify().id(d=>d.name).parentId(d=>d.parent)(network_list)
            this.root=root

            root.x0 = 0;
            root.y0 = 0;
            root.descendants().forEach((d, i) => {
                d.id = i;
                d._children = d.children;
                if (!that.layoutNetwork[d.data.name].expand) d.children = null;
            });
            this.updateTree(root)
        },
        updateTree()
        {
            //console.log("update...",this.focus_ID)
            window.source=source
            const that=this
            // rendering
            if(!this.focus_ID)this.focus_ID=this.root.data.name
            
            let source=this.root
            let found=1
            // find source
            while(source.data.name!=this.focus_ID && found)
            {
                source.children=source._children
                found=0
                for(let i in source._children)
                {
                    const ch=source._children[i]
                    if(this.focus_ID.startsWith(ch.data.name)){source=ch;found=1;break}
                }
            }

            source.children=source._children
            source.descendants().forEach((d, i) => {
                if (!that.layoutNetwork[d.data.name].expand) d.children = null;
                else d.children=d._children
            });
            
            
            window.root=root
            const root=this.root
            window.source=source
            
            const width=750
            const dx=20
            const dy=width/3
            const margin=({top: 10, right: 80, bottom: 10, left: 80})
            const tree=d3.tree().nodeSize([dx, dy])
            const diagonal = d3.linkHorizontal().x(d => d.y).y(d => d.x)
            const svg = d3.select("#navigation-tree")
                //.attr("viewBox", [-margin.left, -margin.top, width, dx])
                .style("font", "10px sans-serif")
                .style("user-select", "none");

            
            
            const gLink = svg.select("g.links")
                .attr("fill", "none")
                .attr("stroke", "#555")
                .attr("stroke-opacity", 0.4)
                .attr("stroke-width", 1.5);
            const gNode = svg.select("g.nodes")
            window.root=root

            const nodes = root.descendants().reverse();
            const links = root.links();
            window.nodes=nodes
            window.links=links

            // Compute the new tree layout.
            tree(root);
            window.root=root

            let left = root;
            let right = root;
            root.eachBefore(node => {
                if (node.x < left.x) left = node;
                if (node.x > right.x) right = node;
            });

            const height = right.x - left.x + margin.top + margin.bottom;

            const container=svg.select("g.container")
            function zoomed({transform}) {
                container.attr("transform", transform);
            }
            svg.call(d3.zoom()
                .extent([[0, 0], [width, height]])
                .scaleExtent([0.3, 3.3])
                .on("zoom", zoomed));

            const transition = svg.transition()
                .delay(this.delay)
                .duration(this.updateDuration)
                .attr("viewBox", [-margin.left+source.y-this.root.y-dy, left.x +source.x-this.root.x - margin.top, width, height])
                //.tween("resize",null)
            
            container.transition(transition).attr("transform","translate(0,0)")
            d3.zoomTransform(container.node()).k=1;
            d3.zoomTransform(container.node()).x=0;
            d3.zoomTransform(container.node()).y=0;


            // Update the nodes…
            const node = gNode.selectAll("g")
                .data(nodes, d => d.id);

            // Enter any new nodes at the parent's previous position.
            const nodeEnter = node.enter().append("g")
                .attr("transform", d => `translate(${source.y0},${source.x0})`)
                .attr("fill-opacity", 0)
                .attr("stroke-opacity", 0)
            
            if(this.canFocusNode)
            {
                nodeEnter.attr("cursor", d=>d._children? "pointer":"default")
                .on("click", (e,d)=> {
                    // console.log(event,d,that,that.focus_ID)
                    if(d._children)
                    {
                        var new_layout=that.layoutNetwork
                        new_layout[d.data.name].expand=!new_layout[d.data.name].expand
                        if(!new_layout[d.data.name].expand)
                        {
                            for(let key in new_layout)
                            {
                                if(key.startsWith(d.data.name))
                                new_layout[key].expand=false
                            }
                        }
                        that.$store.commit('setLayoutInfo', {
                            layoutNetwork: new_layout,
                            focusID: d.data.name,
                            t: Date.now(),
                        });
                        // if(d.data.name==that.focus_ID)that.updateTree()
                    }
                });
            }

            nodeEnter.append("circle")
                .attr("stroke-width", 10)

            nodeEnter.append("text")
                .attr("dy", "0.31em")
                .attr("x", d => d._children ? -6 : 6)
                .attr("text-anchor", d => d._children ? "end" : "start")
                .text(d => d.data.type.length>=30 ? d.data.type.substr(0,27)+"..."  :  d.data.type)
            .clone(true).lower()
                .attr("stroke-linejoin", "round")
                .attr("stroke-width", 3)
                .attr("stroke", "white");

            // Transition nodes to their new position.
            const nodeUpdate = node.merge(nodeEnter).transition(transition).delay(this.updateDuration)
                .attr("transform", d => `translate(${d.y},${d.x})`)
                .attr("fill-opacity", 1)
                .attr("stroke-opacity", 1)
                
            nodeUpdate.select("circle")
                .attr("r", d=> d.data.name==this.focus_ID ? 3.5 : 2.5)
                .attr("fill", d => d._children ? "#555" : "#999")

            //nodeUpdate.selectAll("text")
                //.attr("font-size", d => d.data.name==this.focus_ID ? "16px" : "14px")
                //.attr("font-weight", d => d.data.name==this.focus_ID ? "800" : "400")

            // Transition exiting nodes to the parent's new position.
            const nodeExit = node.exit().transition(transition).delay(this.updateDuration).remove()
                .attr("transform", d => d.parent ? 
                    `translate(${d.parent.y},${d.parent.x})`:
                    `translate(${-dy},${0})`
                )
                .attr("fill-opacity", 0)
                .attr("stroke-opacity", 0);

            // Update the links…
            const link = gLink.selectAll("path")
                .data(links, d => d.target.id);

            // Enter any new links at the parent's previous position.
            const linkEnter = link.enter().append("path")
                .attr("d", d => {
                const o = {x: source.x0, y: source.y0};
                return diagonal({source: o, target: o});
                });

            // Transition links to their new position.
            link.merge(linkEnter).transition(transition).delay(this.updateDuration)
                .attr("d", diagonal);

            // Transition exiting nodes to the parent's new position.
            link.exit().transition(transition).delay(this.updateDuration).remove()
                .attr("d", d => {
                const o = {x: d.source.x, y: d.source.y};
                return diagonal({source: o, target: o});
                });

            // Stash the old positions for transition.
            root.eachBefore(d => {
            d.x0 = d.x;
            d.y0 = d.y;
            });
            
        }
    }
    
}
</script>

<style scoped>
#navigation-tree {
  width: 100%;
  height: 100%;
  margin:10px 0px;
}
g.container
{
    font-family:"sans-serif";
    font-size: 14px;
}
</style>
