<template>
<svg id="navigation-tree">
    <g class="container" v-if="network">
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

export default {
    name: 'navigation',
    computed: {
        ...mapGetters([
            'network'
        ]),
        ...mapState([
            'focusID'
        ])
    },
    props:['canFocusNode'],
    data(){
        return {
            focus_ID:"" ,
            root:undefined,
            ordering:{},
        }
    },
    watch:{
        focus_ID:function(val){
            //console.log("focus_ID changed",val)
            this.updateTree()
        },
        focusID:function(val){
            this.focus_ID=val
        },
        network:function(val,oldval){
            if(val)this.initTree()
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
            for(let key in this.network)
            {
                if(this.network[key].children.length==0)
                {
                    items[key]=this.network[key].inputs.length
                    if(this.network[key].inputs.length==0)ordering.push(key)
                    n++
                }
            }
            for(let i=0;i<n;i++)
            {
                const key=ordering[i]
                this.ordering[key]=i+1
                if(i>=ordering.length)return false
                for(let j in this.network[key].outputs)
                {
                    const chkey=this.network[key].outputs[j]
                    if(--items[chkey]==0)ordering.push(chkey)
                }
            }
            const that=this
            const calcMiddle=(key)=>{
                if(that.ordering[key])return that.ordering[key]
                let ans=0
                for(let ch in that.network[key].children){
                    if(calcMiddle(that.network[key].children[ch]) >ans)
                        ans=that.ordering[that.network[key].children[ch]]
                }
                that.ordering[key]=ans
                return ans
            }
            for(let key in this.network)calcMiddle(key)

            // building tree_list
            window.d3=d3
            window.network=this.network
            let network_list=[]
            for (let key in this.network)
            {
                const id=this.network[key].id
                const fa=this.network[key].parent
                network_list.push({
                    "name":id,
                    "parent":fa,
                    //"attrs":this.network[key].attrs,
                    "type":this.network[key].type,
                    "ordering":this.ordering[id]
                })
            }
            network_list.sort((i,j)=>{
                return i.ordering-j.ordering
            })
            window.network_list=network_list
            const root = d3.stratify().id(d=>d.name).parentId(d=>d.parent)(network_list)
            this.root=root

            root.x0 = 0;
            root.y0 = 0;
            root.descendants().forEach((d, i) => {
                d.id = i;
                d._children = d.children;
                if (d.depth) d.children = null;
            });
            
            this.updateTree(root)
        },
        updateTree(source=undefined)
        {
            //console.log("update...",this.focus_ID)
            window.source=source
            // rendering
            if(!this.focus_ID){
                for(let key in this.network)
                {
                    if(!this.network[key].parent)this.focus_ID=key
                }
            }
            let root=this.root
            let found=1
            // find root
            while(root.data.name!=this.focus_ID && found)
            {
                found=0
                for(let i in root._children)
                {
                    const ch=root._children[i]
                    if(this.focus_ID.startsWith(ch.data.name)){root=ch;found=1;break}
                }
            }

            if(!source)source=root
            root.descendants().forEach(d=>{
                if(this.focus_ID.startsWith(d.data.name))d.children=d._children
                else d.children=null
            })
            root = root.parent || root
            
            root.descendants().forEach(d=>{
                if(this.focus_ID.startsWith(d.data.name))d.children=d._children
                else d.children=null
            })
            
            window.root=root

            //console.log("source: ",source)
            
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
            
            const duration = 500;
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

            const transition = svg.transition()
                .duration(duration)
                //.tween("resize",null)
                .attr("viewBox", [-margin.left+root.y-this.root.y, left.x - margin.top, width, height])
                //.tween("resize", window.ResizeObserver ? null : () => () => svg.dispatch("toggle"));
                

            // Update the nodes…
            const node = gNode.selectAll("g")
                .data(nodes, d => d.id);

            // Enter any new nodes at the parent's previous position.
            const that=this
            const nodeEnter = node.enter().append("g")
                .attr("transform", d => `translate(${source.y0},${source.x0})`)
                .attr("fill-opacity", 0)
                .attr("stroke-opacity", 0)
            
            if(this.canFocusNode)
            {
                nodeEnter.attr("cursor", d=>d._children? "pointer":"default")
                .on("click", (e,d)=> {
                    //console.log(event,d,that,that.focus_ID)
                    if(d._children)
                    {
                        if(d.data.name==that.focus_ID)
                        {
                            that.$store.commit('setFocusID',d.parent.data.name || that.focus_ID)
                        }
                        else
                        {
                            that.$store.commit('setFocusID',d.data.name)
                        }
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
            const nodeUpdate = node.merge(nodeEnter).transition(transition)
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
            const nodeExit = node.exit().transition(transition).remove()
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
            link.merge(linkEnter).transition(transition)
                .attr("d", diagonal);

            // Transition exiting nodes to the parent's new position.
            link.exit().transition(transition).remove()
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
  margin:10px;
}
g.container
{
    font-family:"sans-serif";
    font-size: 14px;
}
</style>
