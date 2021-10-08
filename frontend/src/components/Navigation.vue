<template>
<svg id="navigation-tree">
    <g class="container">
        <g class="links"></g>
        <g class="nodes"></g>
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
    data(){
        return {
            _focusID:undefined,
            ordering:{},
        }
    },
    watch:{
        _focusID(val,val_0){
            console.log("value changed",val)
        }
    },
    created() {
        const that=this
        setTimeout(()=>{
            that.calcTopo();
            that.updateTree()
        },1000)
        setTimeout(()=>{that._focusID="_model/layer3/";},2000)
        setTimeout(()=>{that.updateTree()},3000)
        setTimeout(()=>{that._focusID="_model/layer4/";},4000)
        setTimeout(()=>{that.updateTree()},5000)
    },
    methods:{
        calcTopo()
        {
            console.log(this.network)
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
            for(let key in this.network)console.log(key,calcMiddle(key))
        },
        updateTree(){
            console.log(this.network)
            console.log("..update",this._focusID)
            window.d3=d3
            window.network=this.network
            const width=600

            // building tree
            // required:  the id of the nodes must follow the pattern "fa/fa/fa/fa/id/"

            let network_list=[]

            
            if(!this._focusID){
                for(let key in this.network)
                {
                    if(!this.network[key].parent)this._focusID=key
                }
            }
            let faID=this._focusID.slice(0,this._focusID.slice(0,-1).lastIndexOf('/'))+"/"

            for (let key in this.network)
            {
                const id=this.network[key].id
                const fa=this.network[key].parent
                if(fa==this._focusID || fa==faID)
                {
                    network_list.push(
                    {
                        "id":id,
                        "parent":fa,
                        "attrs":this.network[key].attrs,
                        "ordering":this.ordering[id]
                    })
                }
                else if(id==faID){
                    network_list.push(
                    {
                        "id":id,
                        "parent":undefined,
                        "attrs":this.network[key].attrs,
                        "ordering":this.ordering[id]
                    })
                }
            }
            network_list.sort((i,j)=>{
                return i.ordering-j.ordering
            })
            window.network_list=network_list

            var root=d3.stratify().id(d=>d["id"]).parentId(d=>d["parent"])(network_list)
            const tree = data => {
                const root = d3.hierarchy(data);
                root.dx = 20;
                root.dy = width / (root.height + 1);
                return d3.tree().nodeSize([root.dx, root.dy])(root);
            }
            root=tree(root)
            
            window.root=root

            // rendering
            let x0 = Infinity;
            let x1 = -x0;
            root.each(d => {
                if (d.x > x1) x1 = d.x;
                if (d.x < x0) x0 = d.x;
            });

            const svg = d3.select("#navigation-tree")
                .attr("viewBox", [0, 0, width, x1 - x0 + root.dx * 2]);
            window.svg=svg

            const g = svg.select("g.container")
                .attr("transform", `translate(${root.dy / 3},${root.dx - x0})`)
                
            const reveal = path => path.transition()
                .duration(1000)
                .ease(d3.easeLinear)
                .attrTween("stroke-dasharray", function() {
                    const length = this.getTotalLength();
                    return d3.interpolate(`0,${length}`, `${length},${length}`);
                })                                                                                      
            const link = g.select("g.links")
                .attr("fill", "none")
                .attr("stroke", "#555")
                .attr("stroke-opacity", 0.6)
                .attr("stroke-width", 1)
                .selectAll("path")
                .data(root.links())
                .join("path")
                // .transition()
                // .duration(500)
                .attr("d", d3.linkHorizontal()
                    .x(d => d.y)
                    .y(d => d.x))
                .attr("fill", "none")
                .attr("stroke", "steelblue")
                .attr("stroke-width", 1.5)
                .attr("stroke-miterlimit", "1")
                .attr("stroke-dasharray", "0,1")
                .call(reveal);
            
            const node = g.select("g.nodes")
                .attr("stroke-linejoin", "round")
                .attr("stroke-width", 3)
                .selectAll("g")
                .data(root.descendants())
                .join("g")
                .attr("transform", d => `translate(${d.y},${d.x})`);
            
            node.selectAll("text").remove()
            node.selectAll("circle").remove()

            node.append("circle")
                .attr("fill", d => d.children ? "#555" : "#999")
                .attr("r", 2.5);

            node.append("text")
                .attr("dy", "0.31em")
                .attr("x", d => d.children ? -6 : 6)
                .attr("text-anchor", d => d.children ? "end" : "start")
                //.text(d => d.data.id.slice(d.parent && d.parent.data.id.length,-1))
                //.text(d=> `${d.data.data.attrs.type}(${d.data.data.ordering})`)
                .text(d=> d.data.data.attrs.type)
                .clone(true).lower()
                .attr("stroke", "white");
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
    font-size: 10;
}
</style>
